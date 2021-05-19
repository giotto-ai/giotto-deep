# %%
import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing
import os
from einops import rearrange  # type: ignore
from gtda.homology import WeakAlphaPersistence  # type: ignore
from gtda.plotting import plot_diagram  # type: ignore
from gdeep.create_data import generate_orbit
from gdeep.topology_layers import SmallSetTransformer

# %%
try:
    assert os.path.isdir('./data/ORBIT5K')
except AssertionError:
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    os.mkdir('./data/ORBIT5K')

# If `use_precomputed_dgms` is `False` the ORBIT5K dataset will
# be recomputed, otherwise the ORBIT5K dataset in the folder
# `data/ORBIT5K` will be used
use_precomputed_dgms = True

dgms_filename = os.path.join('data', 'ORBIT5K',
                             'alpha_persistence_diagrams.npy')

if use_precomputed_dgms:
    try:
        assert(os.path.isfile(dgms_filename))
    except AssertionError:
        print('File data/ORBIT5K/alpha_persistence_diagrams.npy',
              ' does not exist.')

# %%

parameters = (2.5, 3.5, 4.0, 4.1, 4.3)  # different classes of orbits
homology_dimensions = (0, 1)

config = {
    'parameters': parameters,
    'num_classes': len(parameters),
    'num_orbits': 1000,
    'num_pts_per_orbit': 1000,
    'homology_dimensions': homology_dimensions,
    'num_homology_dimensions': len(homology_dimensions)
}

if not use_precomputed_dgms:
    # Generate dataset consisting of 5 different orbit types with
    # 1000 sampled data points each.
    # This is the dataset ORBIT5K used in the PersLay paper

    x = np.zeros((
                    config['num_classes'],  # type: ignore
                    config['num_orbits'],
                    config['num_pts_per_orbit'],
                    2
                ))

    # generate dataset
    for cidx, p in enumerate(config['parameters']):  # type: ignore
        for _ in range(config['num_orbits']):  # type: ignore
            x[cidx] = generate_orbit(
                num_pts_per_orbit=config['num_pts_per_orbit'],  # type: ignore
                parameter=p
                )

    # compute weak alpha persistence
    wap = WeakAlphaPersistence(
                        homology_dimensions=config['homology_dimensions'],
                        n_jobs=multiprocessing.cpu_count()
                        )
    # c: class, o: orbit, p: point, d: dimension
    x_stack = rearrange(x, 'c o p d -> (c o) p d')  # stack classes
    diagrams = wap.fit_transform(x_stack)
    # shape: (num_classes * n_samples, n_features, 3)

    diagrams = rearrange(
                            diagrams,
                            '(c o) p d -> c o p d',
                            c=config['num_classes']  # type: ignore
                        )

    # plot sample persistence diagrams
    if(False):
        plot_diagram(diagrams[1, 2])
        plot_diagram(diagrams[2, 2])

    # save dataset
    with open(dgms_filename, 'wb') as f:
        np.save(f, diagrams)
# %%
# load dataset
with open(dgms_filename, 'rb') as f:
    x = np.load(f)

# c: class, o: orbit, p: point in persistence diagram,
# d: coordinates + homology dimension
x = rearrange(
                x,
                'c o p d -> (c o) p d',
                c=config['num_classes']  # type: ignore
            )
# convert homology dimension to one-hot encoding
x = np.concatenate(
    (
        x[:, :, :2],
        (np.eye(config['num_homology_dimensions'])
         [x[:, :, -1].astype(np.int32)]),
    ),
    axis=-1)
# convert from [orbit, sequence_length, feature] to
# [orbit, feature, sequence_length] to fit to the
# input_shape of `SmallSetTransformer`
#x = rearrange(x, 'o s f -> o f s')

# generate labels
y_list = []
for i in range(config['num_classes']):  # type: ignore
    y_list += [i] * config['num_orbits']  # type: ignore

y = np.array(y_list)


# load dataset to PyTorch dataloader

x_tensor = torch.Tensor(x)
y_tensor = torch.Tensor(y)

dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=2 ** 6,
                        num_workers=6)

# initialize SmallSetTransformer model
model = SmallSetTransformer(
                            dim_in=4,
                            dim_out=16,
                            num_heads=4,
                            out_features=config['num_classes']  # type: ignore
                            )

print('model has', model.num_params(), 'trainable parameters.')

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def train(model, num_epochs: int = 10, lr: float = 1e-4,
          verbose: bool = False):
    model = model.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []

    # training loop
    for epoch in range(num_epochs):
        loss_per_epoch = 0
        for x_batch, y_batch in dataloader:
            # transfer to GPU
            x_batch = x_batch.to_device(device)
            y_batch = y_batch.to_device(device)
            loss = criterion(x_batch, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss
        losses.append(loss_per_epoch)
        if verbose:
            print("epoch:", epoch, "loss:", loss_per_epoch)

    return losses
# %%
#model(next(iter(dataloader))[0])
z = next(iter(dataloader))[0]
model(z)
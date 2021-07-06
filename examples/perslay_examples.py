# %%
from typing import List
import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing
import os
from einops import rearrange  # type: ignore
from gtda.homology import WeakAlphaPersistence  # type: ignore
from gtda.plotting import plot_diagram  # type: ignore
from gdeep.topology_layers import ISAB, PMA

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
use_precomputed_dgms = False

dgms_filename = os.path.join('data', 'ORBIT5K',
                             'alpha_persistence_diagrams.npy')
dgms_filename_validation = os.path.join('data', 'ORBIT5K',
                                        'alpha_persistence_diagrams_' +
                                        'validation.npy')

if use_precomputed_dgms:
    try:
        assert(os.path.isfile(dgms_filename))
    except AssertionError:
        print('File data/ORBIT5K/alpha_persistence_diagrams.npy',
              ' does not exist.')
    try:
        assert(os.path.isfile(dgms_filename_validation))
    except AssertionError:
        print('File data/ORBIT5K/alpha_persistence_diagrams.npy',
              ' does not exist.')

# %%
# Create ORBIT5K dataset like in the PersLay paper
parameters = (2.5, 3.5, 4.0, 4.1, 4.3)  # different classes of orbits
homology_dimensions = (0, 1)

config = {
    'parameters': parameters,
    'num_classes': len(parameters),
    'num_orbits': 1_000,  # number of orbits per class
    'num_pts_per_orbit': 1_000,
    'homology_dimensions': homology_dimensions,
    'num_homology_dimensions': len(homology_dimensions),
    'validation_percentage': 100,  # size of validation dataset relative
    # to training
}

if not use_precomputed_dgms:
    for dataset_type in ['train', 'validation']:
        # Generate dataset consisting of 5 different orbit types with
        # 1000 sampled data points each.
        # This is the dataset ORBIT5K used in the PersLay paper
        if dataset_type == 'train':
            num_orbits = config['num_orbits']
        else:
            num_orbits = int(config['num_orbits']  # type: ignore
                             * config['validation_percentage'] / 100)
        x = np.zeros((
                        config['num_classes'],  # type: ignore
                        num_orbits,
                        config['num_pts_per_orbit'],
                        2
                    ))

        # generate dataset
        for cidx, p in enumerate(config['parameters']):  # type: ignore
            x[cidx, :, 0, :] = np.random.rand(num_orbits, 2)  # type: ignore

            for i in range(1, config['num_pts_per_orbit']):  # type: ignore
                x_cur = x[cidx, :, i - 1, 0]
                y_cur = x[cidx, :, i - 1, 1]

                x[cidx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
                x_next = x[cidx, :, i, 0]
                x[cidx, :, i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1

        """
        # old non-parallel version
        for cidx, p in enumerate(config['parameters']):  # type: ignore
            for i in range(config['num_orbits']):  # type: ignore
                x[cidx][i] = generate_orbit(
                    num_pts_per_orbit=config['num_pts_per_orbit'],
                    parameter=p
                    )
        """

        assert(not np.allclose(x[0, 0], x[0, 1]))

        # compute weak alpha persistence
        wap = WeakAlphaPersistence(
                            homology_dimensions=config['homology_dimensions'],
                            n_jobs=multiprocessing.cpu_count()
                            )
        # c: class, o: orbit, p: point, d: dimension
        x_stack = rearrange(x, 'c o p d -> (c o) p d')  # stack classes
        diagrams = wap.fit_transform(x_stack)
        # shape: (num_classes * n_samples, n_features, 3)

        # combine class and orbit dimensions
        diagrams = rearrange(
                                diagrams,
                                '(c o) p d -> c o p d',
                                c=config['num_classes']  # type: ignore
                            )

        # plot sample persistence diagrams for debugging
        if(False):
            plot_diagram(diagrams[1, 2])
            plot_diagram(diagrams[2, 2])

        # save dataset
        if dataset_type == 'train':
            with open(dgms_filename, 'wb') as f:
                np.save(f, diagrams)
        else:
            with open(dgms_filename_validation, 'wb') as f:
                np.save(f, diagrams)
# %%
# load dataset
for dataset_type in ['train', 'validation']:

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
            (np.eye(config['num_homology_dimensions'])  # type: ignore
             [x[:, :, -1].astype(np.int32)]),
        ),
        axis=-1)
    # convert from [orbit, sequence_length, feature] to
    # [orbit, feature, sequence_length] to fit to the
    # input_shape of `SmallSetTransformer`
    # x = rearrange(x, 'o s f -> o f s')

    # generate labels
    y_list = []
    for i in range(config['num_classes']):  # type: ignore
        y_list += [i] * config['num_orbits']  # type: ignore

    y = np.array(y_list)

    # load dataset to PyTorch dataloader

    x_tensor = torch.Tensor(x)
    y_tensor = torch.Tensor(y)

    dataset = TensorDataset(x_tensor, y_tensor)
    if dataset_type == 'train':
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=2 ** 6,
                                num_workers=6)
    else:
        dataloader_validation = DataLoader(dataset,
                                           batch_size=2 ** 6,
                                           num_workers=6)

# %%


# initialize SetTransformer model
class SetTransformer(nn.Module):
    """ Vanilla SetTransformer from
    https://github.com/juho-lee/set_transformer/blob/master/main_pointcloud.py
    """
    def __init__(
        self,
        dim_input=3,  # dimension of input data for each element in the set
        num_outputs=1,
        dim_output=40,  # number of classes
        num_inds=32,  # number of induced points, see  Set Transformer paper
        dim_hidden=128,
        num_heads=4,
        ln=False,  # use layer norm
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, input):
        return self.dec(self.enc(input)).squeeze()


model = SetTransformer(dim_input=4, dim_output=5)


def num_params(model: nn.Module) -> int:
    return sum([parameter.nelement() for parameter in model.parameters()])


print('model has', num_params(model), 'trainable parameters.')  # type: ignore

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# %%
def train(model, num_epochs: int = 10, lr: float = 1e-3,
          verbose: bool = False) -> List[float]:
    """Custom training loop for Set Transformer on the dataset ``

    Args:
        model (nn.Module): Set Transformer model to be trained
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        lr (float, optional): Learning rate for training. Defaults to 1e-3.
        verbose (bool, optional): Print training loss, training accuracy and
            validation if set to True. Defaults to False.

    Returns:
        List[float]: List of training losses
    """
    if use_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    # training loop
    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        for x_batch, y_batch in dataloader:
            # transfer to GPU
            if use_cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            loss = criterion(model(x_batch), y_batch.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
        losses.append(loss_per_epoch)
        if verbose:
            print("epoch:", epoch, "loss:", loss_per_epoch)
            compute_accuracy(model, 'test')
            compute_accuracy(model, 'validation')
    return losses


def compute_accuracy(model, type: str = 'test') -> None:
    correct = 0
    total = 0
    if type == 'test':
        dl = dataloader
    else:
        dl = dataloader_validation

    with torch.no_grad():
        for x_batch, y_batch in dl:
            if use_cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            outputs = model(x_batch).squeeze(1)
            _, predictions = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predictions == y_batch).sum().item()

    print(type.capitalize(),
          'accuracy of the network on the', total,
          'diagrams: %8.2f %%' % (100 * correct / total)
          )


# %%
train(model, num_epochs=500, verbose=True)

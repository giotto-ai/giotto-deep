# %%
%load_ext autoreload
%autoreload 2
import random
import time
import matplotlib.pyplot as plt
from typing import List
from attrdict import AttrDict
import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing
import os
from einops import rearrange  # type: ignore
from gtda.homology import WeakAlphaPersistence  # type: ignore
from gtda.plotting import plot_diagram  # type: ignore
from gdeep.topology_layers import SelfAttentionSetTransformer, train_vec, sam_train

#%%

parameters = (2.5, 3.5, 4.0, 4.1, 4.3)  # different classes of orbits
homology_dimensions = (0, 1)
n_points = 1_000  # should be divisible by len(parameters)
k = int(n_points / 1000)

config = AttrDict({
            'n_points': n_points,
            'dataset_name': 'ORBIT'+ str(k) + 'K',
            'parameters': parameters,
            'num_classes': len(parameters),
            'num_orbits': int(n_points / len(parameters)),  # number of orbits per class
            'num_pts_per_orbit': int(n_points / 5),
            'homology_dimensions': homology_dimensions,
            'num_homology_dimensions': len(homology_dimensions),
            'validation_percentage': 100,  # size of validation dataset relative
            # to training if 100 the train and validation datasets have the
            # same size
            'use_precomputed_dgms': False,
         })

# %%
try:
    assert os.path.isdir(
        os.path.join('./data', config.dataset_name)
        )
except AssertionError:
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    os.mkdir(
        os.path.join('./data', config.dataset_name)
    )

# If `use_precomputed_dgms` is `False` the ORBIT5K dataset will
# be recomputed, otherwise the ORBIT5K dataset in the folder
# `data/ORBIT5K` will be used

dgms_filename_train = os.path.join('data', config.dataset_name,
                             'alpha_persistence_diagrams.npy')
dgms_filename_validation = os.path.join('data', config.dataset_name,
                                        'alpha_persistence_diagrams_' +
                                        'validation.npy')

if config.use_precomputed_dgms:
    try:
        assert(os.path.isfile(dgms_filename_train))
    except AssertionError:
        print('File data/' + config.dataset_name +
              '/alpha_persistence_diagrams.npy',
              ' does not exist.')
    try:
        assert(os.path.isfile(dgms_filename_validation))
    except AssertionError:
        print('File data/' + config.dataset_name +
              '/alpha_persistence_diagrams.npy',
              ' does not exist.')

# %%
# Create ORBIT5K dataset like in the PersLay paper

if not config.use_precomputed_dgms:
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
            with open(dgms_filename_train, 'wb') as f:
                np.save(f, diagrams)
        else:
            with open(dgms_filename_validation, 'wb') as f:
                np.save(f, diagrams)
# %%
# load dataset
for dataset_type in ['train', 'validation']:
    if dataset_type == 'train':
        dgms_filename = dgms_filename_train
    else:
        dgms_filename = dgms_filename_validation

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
    print(dataset_type, x_tensor[0])
    y_tensor = torch.Tensor(y).long()

    dataset = TensorDataset(x_tensor, y_tensor)
    if dataset_type == 'train':
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=2 ** 6)
    else:
        dataloader_validation = DataLoader(dataset,
                                           batch_size=2 ** 6)

# %%


# initialize SetTransformer model

model = SelfAttentionSetTransformer(dim_input=4, dim_output=5)


def num_params(model: nn.Module) -> int:
    return sum([parameter.nelement() for parameter in model.parameters()])


print('model has', num_params(model), 'trainable parameters.')  # type: ignore

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# %%
# Define Model architecture for the graph classifier

num_runs = 3
n_epochs = 200


for i in range(num_runs):
    print("Test run number", i + 1, "of", num_runs)
    lr = random.choice([1e-2, 1e-3, 1e-4])
    ln = random.choice([True])  # LayerNorm in Set Transformer
    use_regularization = random.choice([False])  # Use L2-regularization
    use_induced_attention = random.choice([False])  # use trainable query vector instead of
    # self-attention; use induced attention for large sets because of the
    # quadratic scaling of self-attention.
    # the class with more points
    # only use the persistence diagrams as features not the spectral features
    optimizer = lambda params: torch.optim.Adam(params, lr=lr)  # noqa: E731
    dim_hidden = random.choice([64, 128, 256, 512])
    dim_output = random.choice([20, 50, 100])
    num_heads = random.choice([4, 8, 16, 32])
    num_layers = random.choice([1, 2, 4, 8])
    use_sam = random.choice([False])

    print(f"""lr={lr}\nregularization={use_regularization}\nuse_induced_attention={use_induced_attention}\nln={ln}\ndim_output={dim_output}\nnum_heads={num_heads}\ndim_hidden={dim_hidden}""")

    # define graph classifier
    gc = SelfAttentionSetTransformer(
            dim_input=4,
            num_outputs=1,
            dim_output=config.num_classes,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
            n_layers=num_layers
            )

    if use_sam:
        train_fct = sam_train
    else:
        train_fct = train_vec

    # train the model and return losses and accuracies information
    tic = time.perf_counter()
    (losses,
     val_losses,
     train_accuracies,
     val_accuracies) = train_fct(
                                gc,
                                dataloader,
                                dataloader_validation,
                                lr=lr,
                                verbose=True,
                                num_epochs=n_epochs,
                                use_cuda=use_cuda,
                                use_regularization=use_regularization,
                                optimizer=optimizer
                                )
    toc = time.perf_counter()
    print(f"Trained model for {n_epochs} in {toc - tic:0.4f} seconds")
    del gc
    # plot losses
    plt.plot(losses, label='train_loss')
    plt.plot([x for x in val_losses], label='val_loss')
    plt.legend()
    plt.title("Losses " + config.dataset_name + " extended persistence features only")

    plt.savefig("losses.png")
    plt.show()

    # plot accuracies
    plt.plot(train_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.legend()
    plt.title("Accuracies " + config.dataset_name + " extended persistence features only")

    plt.savefig("accuracies.png")
    plt.show()
    # plot metadata
    plt.text(0.2, 0.5,
             f"""\nlr={lr}\nuse_regularization={use_regularization}\nuse_induced_attention={use_induced_attention}\nln={ln}\ndim_output={dim_output}\nnum_heads={num_heads}\ndim_hidden={dim_hidden}""",
             fontsize='xx-large')
    plt.axis('off')

    plt.savefig("metadata.png")
    plt.show()

    import sys
    from PIL import Image

    images = [Image.open(x) for x in ['losses.png', 'accuracies.png', 'metadata.png']]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    max_val_acc = '{:2.2%}'.format(max(val_accuracies)/100)
    new_im.save(f"""Benchmark_PersFormer_graph/{config.dataset_name}_Benchmark/max_val_acc{{max_val_acc}}_lr_{lr}_{use_regularization}_{use_induced_attention}_{ln}_{dim_output}_{num_heads}_{dim_hidden}_epochs_{n_epochs}_run_{i}.png""")


# %%

# %%

# %%

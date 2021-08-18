# %%
%load_ext autoreload
%autoreload 2
import random
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)
import matplotlib.pyplot as plt
from typing import List
from attrdict import AttrDict
import numpy as np  # type: ignore
import torch
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange  # type: ignore
import os
from gdeep.topology_layers import SelfAttentionSetTransformer, train_vec,\
    sam_train
from gdeep.create_data import generate_orbit_parallel, create_pd_orbits,\
    convert_pd_orbits_to_tensor

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
            'num_orbits': int(n_points / len(parameters)),
            # number of orbits per class
            'num_pts_per_orbit': int(n_points / 5),
            'homology_dimensions': homology_dimensions,
            'num_homology_dimensions': len(homology_dimensions),
            'validation_percentage': 100,  # size of validation dataset
            # relative to training if 100 the train and validation datasets
            # have the same size
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

# Load the ORBIT5K dataset if it exists and if config.use_precomputed_dgms
# is set to `True`
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
        orbits = generate_orbit_parallel(
            num_classes = config.num_classes,
            num_orbits=num_orbits,
            num_pts_per_orbit=config.num_pts_per_orbit,
            parameters=config.parameters,
        )


        diagrams = create_pd_orbits(
            orbits,
            num_classes = config.num_classes,
            homology_dimensions=config.homology_dimensions,
        )


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

    x_tensor, y_tensor = convert_pd_orbits_to_tensor(
        diagrams=x,
        num_classes=config['num_classes'],
        num_orbits=config['num_orbits'],
        num_homology_dimensions=config['num_homology_dimensions'], 
    )

    dataset = TensorDataset(x_tensor, y_tensor)
    if dataset_type == 'train':
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=2 ** 3)
    else:
        dataloader_validation = DataLoader(dataset,
                                           batch_size=2 ** 3)

# %%

# Load CUDA for PyTorch if  available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Specify the random parameter search procedure
num_runs = 2
n_epochs = 1


for i in range(num_runs):
    print("Test run number", i + 1, "of", num_runs)

    # Randomly choose the hyper-parameters
    config_run = AttrDict({})
    config_run.lr = random.choice([1e-2, 1e-3, 1e-4])
    config_run.ln = random.choice([True])  # LayerNorm in Set Transformer
    config_run.use_regularization = random.choice([False, True])
    # Use L2-regularization
    config_run.use_induced_attention = random.choice([False])
    # use trainable query vector instead of
    # self-attention; use induced attention for large sets because of the
    # quadratic scaling of self-attention.
    # the class with more points
    # only use the persistence diagrams as features not the spectral features
    config_run.optimizer = lambda params: torch.optim.Adam(params,
                                                           lr=config_run.lr)
    config_run.dim_hidden = random.choice([64, 128, 256, 512])
    config_run.dim_output = random.choice([20, 50, 100])
    config_run.num_heads = random.choice([4, 8, 16])
    config_run.num_layers = random.choice([1, 2, 3])
    config_run.use_sam = random.choice([False])


    # initialize set transformer model
    gc = SelfAttentionSetTransformer(
            dim_input=4,
            num_outputs=1,
            dim_output=config.num_classes,
            num_heads=config_run.num_heads,
            dim_hidden=config_run.dim_hidden,
            n_layers=config_run.num_layers
            )

    # Choose the training method
    if config_run.use_sam:
        train_fct = sam_train
    else:
        train_fct = train_vec

    # train the model and return losses and accuracies information
    tic = time.perf_counter()
    
    #Print number of trainable parameters
    print(f"number of trainable parameters: {gc.num_params}")

    # Print the model configuration
    pp.pprint(config_run)

    # Trained the model and return loss and accuracy information
    (losses,
     val_losses,
     train_accuracies,
     val_accuracies) = train_fct(
                                gc,
                                dataloader,
                                dataloader_validation,
                                lr=config_run.lr,
                                verbose=True,
                                num_epochs=n_epochs,
                                use_cuda=use_cuda,
                                use_regularization=config_run\
                                    .use_regularization,
                                optimizer=config_run.optimizer
                                )
    toc = time.perf_counter()
    print(f"Trained model for {n_epochs} in {toc - tic:0.4f} seconds")
    del gc
    # plot losses
    
# %%
def save_run_summary(losses, val_losses, config)->None:
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
             f"""\nlr={lr}\nuse_regularization={use_regularization}\nuse_induced_attention={use_induced_attention}\nln={ln}\ndim_output={dim_output}\nnum_heads={num_heads}\ndim_hidden={dim_hidden}\nnum_layers={num_layers}""",
             fontsize='x-large')
    plt.axis('off')

    plt.savefig("metadata.png")
    plt.show()

    image_files = ['losses.png', 'accuracies.png', 'metadata.png']
    images = [Image.open(x) for x in image_files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    max_val_acc = '{:2.2%}'.format(max(val_accuracies)/100)
    new_im.save(f"""Benchmark_PersFormer_graph/{config.dataset_name}_Benchmark/max_val_acc{max_val_acc}_lr_{lr}_{use_regularization}_{use_induced_attention}_{ln}_{dim_output}_{num_heads}_{dim_hidden}_{num_layers}_epochs_{n_epochs}_run_{i}.png""")

    # delete temporary images
    for image in image_files:
        os.remove(image)

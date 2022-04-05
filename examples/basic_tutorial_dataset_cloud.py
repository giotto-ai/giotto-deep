# %%
from gdeep.utility.utils import is_notebook

if is_notebook:
    # Autoreload modules
    from IPython import get_ipython  # type: ignore
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
# %%
import torch
import os
from os import remove
from os.path import join

from gdeep.data import DlBuilderFromDataCloud, DatasetCloud
from gdeep.utility.utils import get_checksum


# %%
folder = "examples/data/DatasetCloud/SmallDataset"
# Compute checksums for all files folder of type 'pt' and print them
def print_checksums(folder):
    """Prints the checksums of all files in folder of type 'pt'.
    
    Parameters
    ----------
    folder : str
        The path to the folder.
    
    Returns
    -------
    None
        This function does not return anything.
    """
    for file in os.listdir(folder):
        if file.endswith(".pt"):
            print(file, ":", get_checksum(join(folder, file)))

print_checksums(folder)

# %% [markdown]
# ## Using the Dataset Cloud to train topological models
# In this tutorial we will use the our custom datasets storage on
# [Google Cloud Datastore](https://cloud.google.com/datastore/) to
# to store and load our datasets.
# The dataset cloud storage contain a variety of topological datasets that
# can be easily used in GDeep.

# %%
# To see all publically available datasets, please use the following command:
DatasetCloud("").get_existing_dataset()

# %% [markdown]
# ## Uploading and downloading datasets from the cloud
# Using datasets from the cloud is very easy. The datasets are publicly
# available and can be downloaded from the cloud without any registration.
#
# To upload a dataset to the cloud, you have to have a Google Cloud API key
# to the Google Cloud Datastore bucket. If you are interested uploading your
# own dataset, please contact us at
# [raphael.reinauer@epfl.ch](mailto:raphael.reinauer@epfl.ch).
# %%
def create_and_upload_dataset():
    """The method above creates a dataset with random data and labels,
    saves it locally as pickled files, and then uploads it to the Cloud.
    The dataset is then deleted from the local machine.
    
    Returns
    -------
    None
        This function does not return anything.
    """
    # Generate a dataset
    # You do not have to do that if you already have a pickled dataset
    size_dataset = 100
    input_dim = 5
    num_labels = 2
    data = torch.rand(size_dataset, input_dim)
    labels = torch.randint(0, num_labels, (size_dataset,)).long()

    # pickle data and labels
    data_filename = 'tmp_data.pt'
    labels_filename = 'tmp_labels.pt'
    torch.save(data, data_filename)
    torch.save(labels, labels_filename)

    ## Upload dataset to Cloud
    dataset_name = "SmallDataset2"
    dataset_cloud = DatasetCloud(dataset_name)

    # Specify the metadata of the dataset
    dataset_cloud._add_metadata(
        name=dataset_name,
        input_size=(input_dim,),
        size_dataset=size_dataset,
        num_labels=num_labels,
        data_type="tabular",
        data_format="pytorch_tensor",
    )

    # upload dataset to Cloud
    dataset_cloud._upload(data_filename, labels_filename)

    # remove the labels and data files
    # Warning: Only do this if you do want the local dataset to be deleted!
    remove(data_filename)
    remove(labels_filename)
    
create_and_upload_dataset()
# %% [markdown]
# ## Using the Dataset Cloud to train topological model
# The datasets in the cloud are automatically downloaded and used by GDeep.
# Only specify the dataset name and the path you want to save the model.
# %%
# Create dataloaders from data cloud
# If you don't know what datasets exist in the cloud, just use an empty
# ´dataset_name´ and then the error message will display all available datasets 
dataset_name = "MutagDataset"
download_directory = join("data", "DatasetCloud")

dl_cloud_builder = DlBuilderFromDataCloud(dataset_name,
                                   download_directory)

# You can display the metadata of the dataset
print(dl_cloud_builder.get_metadata())

# create the dataset from the downloaded dataset
train_dataloader, val_dataloader, test_dataloader = \
    dl_cloud_builder.build_dataloaders(batch_size=10)

del train_dataloader, val_dataloader, test_dataloader

# %% [markdown]
# Now you can train a model on the dataset using the created dataloaders.

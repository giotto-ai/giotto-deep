# %% 
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# %%
import torch
import os
from os import remove
from os.path import join

from gdeep.data import DlBuilderFromDataCloud, DatasetCloud

from gdeep.search import PersformerHyperparameterSearch
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
    dataset_name = "SmallDataset"
    dataset_cloud = DatasetCloud(dataset_name)

    # Specify the metadata of the dataset
    dataset_cloud.add_metadata(
        name=dataset_name,
        input_size=(input_dim,),
        size_dataset=size_dataset,
        num_labels=num_labels,
        data_type="tabular",
        data_format="pytorch_tensor"
    )

    # upload dataset to Cloud
    dataset_cloud.upload(data_filename, labels_filename)

    # remove the labels and data files
    # Warning: Only do this if you do want the local dataset to be deleted!
    remove(data_filename)
    remove(labels_filename)
    
create_and_upload_dataset()

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
train_dataloader, val_dataloader, test_dataloader = dl_cloud_builder.build_dataloaders(batch_size=10)

del train_dataloader, val_dataloader, test_dataloader

# %%

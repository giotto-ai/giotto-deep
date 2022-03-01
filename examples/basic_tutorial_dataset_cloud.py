# %% 
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# %%
import torch
from gdeep.data import DlBuilderFromDataCloud
from os import remove
from os.path import join


# %%

def create_and_upload_dataset():
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
dataset_name = "SmallDataset"
download_directory = join("data", "DatasetCloud")

dl_cloud_builder = DlBuilderFromDataCloud(dataset_name,
                                   download_directory)

train_dataloader, val_dataloader, test_dataloader = dl_cloud_builder.build_dataloaders()
# %%
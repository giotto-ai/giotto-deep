from gdeep.data import DatasetCloud, dataset_cloud

import hashlib
import logging
import os
from os import remove, environ
from os.path import join, exists
from shutil import rmtree

import numpy as np  # type: ignore
import torch

from gdeep.utility.utils import get_checksum

LOGGER = logging.getLogger(__name__)


# Test public access for downloading datasets
def test_public_access():
    # Download a small dataset from Google Cloud Storage
    dataset = "SmallDataset"
    download_directory = join("examples", "data", "DatasetCloud", "Tmp")
    
    # Remove download directory recursively if it exists
    if exists(download_directory):
        rmtree(download_directory)
    
    # Create download directory
    os.makedirs(download_directory, exist_ok=False)
    
    
    dataset_cloud = DatasetCloud(dataset,
                                 download_directory=download_directory,
                                 use_public_access=True)
    dataset_cloud.download()
    
    # Check if the downloaded files (metadata.json, data.json, labels.json)
    # are correct
    checksums = {"data.pt": "2ef68a718e29134cbcbf46c9592f6168",
                 "labels.pt": "d71992425033c6bf449d175db146a423",
                 }
    
    for file in checksums.keys():
        assert get_checksum(join(download_directory,
                                       dataset,
                                       file)) == \
            checksums[file], "File {} is corrupted.".format(file)
    # Recursively remove download directory
    rmtree(download_directory)


    
           
def test_get_dataset_list():
    # Download directory will not be used as well ass the dataset
    # It's only used for initialization of the DatasetCloud object
    download_directory = ""
    dataset_cloud = DatasetCloud("SmallDataset",
                                 download_directory=download_directory,
                                 use_public_access=True)
    dataset_list = dataset_cloud.get_existing_datasets()
    assert len(dataset_list) > 0, "Dataset list is empty."
    assert "SmallDataset" in dataset_list,\
        "Dataset list does not contain the dataset."
        
    # Test if the list does not contain duplicates
    assert len(dataset_list) == len(set(dataset_list)),\
        "Dataset list contains duplicates."




if "GOOGLE_APPLICATION_CREDENTIALS" in dict(environ):
    def test_update_dataset_list():
        # Create DatasetCloud object
        dataset_cloud = DatasetCloud("", use_public_access=False)
        # Update the dataset list
        dataset_cloud._update_dataset_list()
        
    
    def test_upload_and_download():
        for data_format in ['pytorch_tensor', 'numpy_array']:
            download_directory = join("examples", "data", "DatasetCloud")
            # Generate a dataset
            # You don't have to do that if you already have a pickled dataset
            size_dataset = 100
            input_dim = 5
            num_labels = 2
            
            if data_format == 'pytorch_tensor':
                data = torch.rand(size_dataset, input_dim)
                labels = torch.randint(0, num_labels, (size_dataset,)).long()

                # pickle data and labels
                data_filename = 'tmp_data.pt'
                labels_filename = 'tmp_labels.pt'
                torch.save(data, data_filename)
                torch.save(labels, labels_filename)
            elif data_format == 'numpy_array':
                data = np.random.rand(size_dataset, input_dim)
                labels = np.random.randint(0, num_labels, (size_dataset,),
                                           dtype=np.long)

                # pickle data and labels
                data_filename = 'tmp_data.npy'
                labels_filename = 'tmp_labels.npy'
                np.save(data_filename, data)
                np.save(labels_filename, labels)
            else:
                raise ValueError(f"Unknown data format: {data_format}")

            ## Upload dataset to Cloud
            dataset_name = "TmpSmallDataset"
            dataset_cloud = DatasetCloud(dataset_name,
                                        download_directory=download_directory,
                                        use_public_access=False)

            # Specify the metadata of the dataset
            dataset_cloud._add_metadata(
                name=dataset_name,
                size_dataset=size_dataset,
                input_size=(input_dim,),
                num_labels=num_labels,
                data_type="tabular",
                data_format=data_format,
            )

            # upload dataset to Cloud
            dataset_cloud._upload(data_filename, labels_filename)

            # download dataset from Cloud to ´example/data/DataCloud/SmallDataset/´
            dataset_cloud.download()

            # remove created blob
            dataset_cloud._data_cloud.delete_blobs(dataset_name)

            # check whether downloaded dataset is the same as the original dataset
            if data_format == 'pytorch_tensor':
                downloaded_files = ["data.pt", "labels.pt", "metadata.json"]
            elif data_format == 'numpy_array':
                downloaded_files = ["data.npy", "labels.npy", "metadata.json"]
            for file in downloaded_files:
                hash_original = get_checksum('tmp_' + file)
                path_downloaded_file = join(download_directory, dataset_name, file)
                hash_downloaded = get_checksum(path_downloaded_file)
                assert hash_original == hash_downloaded, \
                    "Original and downloaded files do not match."
                

            # remove the labels and data files
            remove(data_filename)
            remove(labels_filename)

            # remove the downloaded dataset
            rmtree(join(download_directory, dataset_name))

            # remove the metadata file
            # will get deleted automatically when dataset_cloud is out of scope.
            del dataset_cloud
            

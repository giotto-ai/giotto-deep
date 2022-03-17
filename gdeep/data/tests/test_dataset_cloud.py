from gdeep.data import DatasetCloud
import logging
import pytest
import torch
import numpy as np
from os import remove
from os.path import join
import hashlib
from shutil import rmtree
from google.auth.exceptions import DefaultCredentialsError  # type: ignore
import time

LOGGER = logging.getLogger(__name__)

def credentials_error_logging(func):
    def inner():
        try:
            func()
        except DefaultCredentialsError:
            LOGGER.warning("GCP credentials failed.")
    return inner

def file_as_bytes(file):
    """Returns a bytes object representing the file

    Args:
        file (str): File to read.

    Returns:
        _type_: Byte object
    """
    with file:
        return file.read()

@credentials_error_logging
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
            labels = np.random.randint(0, num_labels, (size_dataset,), dtype=np.long)

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
                                    download_directory=download_directory)

        # Specify the metadata of the dataset
        dataset_cloud.add_metadata(
            name=dataset_name,
            size_dataset=size_dataset,
            input_size=(input_dim,),
            num_labels=num_labels,
            data_type="tabular",
            data_format=data_format,
        )

        # upload dataset to Cloud
        dataset_cloud.upload(data_filename, labels_filename)

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
            hash_original = hashlib.md5(file_as_bytes(open('tmp_' + file, 'rb'))).hexdigest()
            path_downloaded_file = join(download_directory, dataset_name, file)
            hash_downloaded = hashlib.md5(file_as_bytes(open(path_downloaded_file, 'rb'))).hexdigest()
            assert hash_original == hash_downloaded, "Original and downloaded files do not match."
            

        # remove the labels and data files
        remove(data_filename)
        remove(labels_filename)

        # remove the downloaded dataset
        rmtree(join(download_directory, dataset_name))

        # remove the metadata file
        # will get deleted automatically when dataset_cloud is out of scope.
        del dataset_cloud
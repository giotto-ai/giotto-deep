import json
import os
import shutil
from os.path import join
from typing import Any, Dict, Tuple, TypeVar, Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset_form_array import FromArray
from .dataset_cloud import DatasetCloud
from .base_dataloaders import DataLoaderBuilder
from .base_dataloaders import AbstractDataLoaderBuilder

Tensor = torch.Tensor
T = TypeVar('T')


class DlBuilderFromDataCloud(AbstractDataLoaderBuilder):
    """Class that loads data from Google Cloud Storage
    
    This class is useful to build dataloaders from a dataset stored in
    the GDeep Dataset Cloud on Google Cloud Storage.

    The constructor takes the name of a dataset as a string, and a string
    for the download directory. The constructor will download the dataset
    to the download directory. The dataset is downloaded in the version
    used by Datasets Cloud, which may be different from the version
    used by the dataset's original developers.
    
    Args:
        dataset_name (str):
            The name of the dataset.
        download_dir (str):
            The directory where the dataset will be downloaded.
        use_public_access (bool):
            Whether to use public access. If you want
            to use the Google Cloud Storage API, you must set this to True.
            Please make sure you have the appropriate credentials.
        path_to_credentials (str):
            Path to the credentials file.
            Only used if public_access is False and credentials are not
            provided. Defaults to None.
        

    Returns:
        torch.utils.data.DataLoader: The dataloader for the dataset.

    Raises:
        ValueError:
            If the dataset_name is not a valid dataset that exists
            in Datasets Cloud.
        ValueError:
            If the download_directory is not a valid directory.
    """

    def __init__(self,
                 dataset_name: str,
                 download_directory: str,
                 use_public_access: bool = True,
                 path_to_credentials: Union[None, str] = None,
                 ):
        self.dataset_name = dataset_name
        self.download_directory = download_directory

        # Download the dataset if it does not exist
        self.download_directory

        self._download_dataset(use_public_access=use_public_access,
                               path_to_credentials=path_to_credentials)

        self.dl_builder = None

        # Load the metadata of the dataset
        with open(join(self.download_directory, self.dataset_name,
                       "metadata.json")) as f:
            self.dataset_metadata = json.load(f)

        # Load the data and labels of the dataset
        if self.dataset_metadata['data_type'] == 'tabular':
            if self.dataset_metadata['data_format'] == 'pytorch_tensor':
                data = torch.load(join(self.download_directory,
                                       self.dataset_name, "data.pt"))
                labels = torch.load(join(self.download_directory,
                                         self.dataset_name, "labels.pt"))

                self.dl_builder = DataLoaderBuilder([FromArray(data, labels), ])
            elif self.dataset_metadata['data_format'] == 'numpy_array':
                data = np.load(join(self.download_directory,
                                    self.dataset_name, "data.npy"))
                labels = np.load(join(self.download_directory,
                                      self.dataset_name, "labels.npy"))
                self.dl_builder = DataLoaderBuilder([FromArray(data, labels), ])
            else:
                raise ValueError(f"Data format {self.dataset_metadata['data_format']}" +
                                 "is not yet supported.")
        else:
            raise ValueError(f"Dataset type {self.dataset_metadata['data_type']} is not yet supported.")

    def _download_dataset(self,
                          path_to_credentials: Union[None, str] = None,
                          use_public_access: bool = True, ) -> None:
        """Only download if the download directory does not exist already
        and if download directory contains at least three files (metadata,
        data, labels).
        
        Args:
            path_to_credentials (str):
                Path to the credentials file.
            use_public_access (bool):
                Whether to use public access. If you want
                to use the Google Cloud Storage API, you must set this to True.
                
        Returns:
            None
        """
        if (not os.path.isdir(join(self.download_directory, self.dataset_name))
                or len(os.listdir(join(self.download_directory,
                                       self.dataset_name))) < 3):
            # Delete the download directory if it exists but does not contain
            # the wanted number of files
            if (os.path.isdir(join(self.download_directory, self.dataset_name))
                    and
                    len(os.listdir(join(self.download_directory,
                                        self.dataset_name))) < 3):  # type: ignore
                print("Deleting the download directory because it does " +
                      "not contain the dataset")
                shutil.rmtree(self.download_directory, ignore_errors=True)

            print("Downloading dataset {} to {}" \
                  .format(self.dataset_name, self.download_directory))
            dataset_cloud = DatasetCloud(self.dataset_name,
                                         download_directory=self.download_directory,
                                         path_to_credentials=path_to_credentials,
                                         use_public_access=use_public_access,
                                         )
            dataset_cloud.download()
            del dataset_cloud
        else:
            print("Dataset '%s' already downloaded" % self.dataset_name)

    def get_metadata(self) -> Dict[str, Any]:
        """ Returns the metadata of the dataset.
        
        Returns:
            Dict[str, Any]:
                The metadata of the dataset.
        """
        return self.dataset_metadata

    def build(self, tuple_of_kwargs: List[Dict[str, Any]]) \
            -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Builds the dataloaders for the dataset.
        
        Args:
            **tuple_of_kwargs: Arguments for the dataloader builder.
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]:
                The dataloaders for the dataset (train, validation, test).
        """
        return self.dl_builder.build(tuple_of_kwargs)  # type: ignore

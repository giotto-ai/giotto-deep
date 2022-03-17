from ._data_cloud import _DataCloud
from typing import Union, Tuple
from os import remove
from os.path import join, split, exists
import json

class DatasetCloud():
    """DatasetCloud class to handle the download and upload of datasets to the DataCloud.

    Args:
        dataset_name (str): Name of the dataset to be downloaded or uploaded.
        bucket_name (str, optional): Name of the bucket in the DataCloud. Defaults to "adversarial_attack".
        download_directory (Union[None, str], optional): Directory where the dataset will be downloaded to. Defaults to None.
        """
    def __init__(self,
             dataset_name: str,
             bucket_name: str = "adversarial_attack",
             download_directory: Union[None, str] = None,
             ):
        self.name = dataset_name
        self.path_metadata = None
        if download_directory is None:
            self.download_directory = join('examples', 'data', 'DataCloud')
        else:
            self.download_directory = download_directory
        self._data_cloud = _DataCloud(bucket_name=bucket_name,
                                    download_directory = self.download_directory)
        
    def __del__(self):
        """This function deletes the metadata file if it exists.

        Returns:
            None
        """
        if not self.path_metadata == None:
            remove(self.path_metadata)
            
    def download(self):
        """Download a dataset from the DataCloud.

        Raises:
            ValueError: Dataset does not exits in cloud.
        """
        # List of existing datasets in the cloud.
        existing_datasets = set([blob.name.split('/')[0] for blob in self._data_cloud.bucket.list_blobs()\
            if blob.name != "giotto-deep-big.png"])
        if not self.name in existing_datasets:
            raise ValueError(f"Dataset {self.name} does not exist in the cloud.\n Available datasets are: {existing_datasets}.")
        if exists(join(self.download_directory, self.name)):
            print(f"Dataset {self.name} already exists in the download directory.")
        self._data_cloud.download_folder(self.name + '/')

    @staticmethod
    def _get_filetype(path: str) -> str:
        """Returns the file extension from a given path.
    
        Args:
            path: A string path.
        
        Returns:
            The file extension from the given path.
        
        Raises:
            None.
        """
    
        return path.split('.')[-1]
    
    def _upload_data(self,
                path: str,) -> None:
        """Uploads the label files to a Cloud Storage bucket.

        Args:
        path: The path to the label file.

        Raises:
        ValueError: If the file type is not supported."""
        filetype = DatasetCloud._get_filetype(path)
        assert filetype in ('pt', 'npy'), "File type not supported."
        self._data_cloud.upload_file(path, self.metadata['name'] + '/data.' + filetype)  # type: ignore
    
    
    def _upload_label(self,
                 path: str,) -> None:
        """Uploads a set of labels to a remote dataset.

        Args:
            path: the path to the labels file.

        Returns:
            None

        Raises:
            AssertionError: if the path is not valid or the filetype is not supported.
        """
        filetype = DatasetCloud._get_filetype(path)
        assert filetype in ('pt', 'npy'), "File type not supported."
        self._data_cloud.upload_file(path, self.metadata['name'] + '/labels.' + filetype)  # type: ignore
    
    def _upload_metadata(self,
                        path: Union[str, None]=None):
        """Uploads the metadata dictionary to the location specified in the metadata. The metadata dictionary is generated using create_metadata.
        
        Args:
            path (str): The path to the data cloud folder. If none, path will be set to the default path.
            
        Raises:
            Exception: If no metadata exists, an exception will be raised."""
        if self.metadata == None:
            raise Exception("No metadata to upload. Please create metadata using create_metadata.")  #NOSONAR
        self._data_cloud.upload_file(path, self.metadata['name'] + '/' + 'metadata.json')  # type: ignore

    def add_metadata(self,
                     size_dataset: int,
                     input_size: Tuple[int, ...],
                     num_labels: Union[None, int] = None,
                     data_type: str = "tabular",
                     task_type: str = "classification",
                     name: Union[None, str]=None,
                     data_format: Union[None, str]=None,
                     comment: Union[None, str]=None,
                     ):
        """This function accepts various metadata for the dataset and stores it in a temporary JSON file.

        Args:
            size_dataset (int): The size of the dataset (in terms of the number of samples).
            input_size (Tuple[int, ...]): The size of each sample in the dataset.
            num_labels (Union[None, int]): The number of classes in the dataset.
            data_type (str): The type of data in the dataset.
            task_type (str): The task type of the dataset.
            name (Union[None, str]): The name of the dataset.
            data_format (Union[None, str]): The format of the data in the dataset.
            comment (Union[None, str]): A comment describing the dataset.

        Returns:
            None
        """
        if name is None:
            name = self.name
        if data_format is None:
            data_format = "pytorch_tensor"
        self.path_metadata = "tmp_metadata.json"  # type: ignore
        self.metadata = {'name': name,
                    'size': size_dataset,
                    'input_size': input_size,
                    'num_labels': num_labels,
                    'task_type': task_type,
                    'data_type': data_type,
                    'data_format': data_format,
                    'comment': comment
                    }
        with open(self.path_metadata, "w") as f:  # type: ignore
            json.dump(self.metadata, f, sort_keys=True, indent=4)
        
    def upload(self,
                       path_data: str,
                       path_label: str,
                       path_metadata: Union[str, None] = None,
                       ):
        """Uploads a dataset to the cloud.

        Args:
            path_data (str): Path to the data files.
            path_label (str): Path to the label file.
            path_metadata (Optional[str]): Path to the metadata file.

        Raises:
            ValueError: If the dataset already exists in the cloud."""
        # List of existing datasets in the cloud.
        existing_datasets = set([blob.name.split('/')[0] for blob in self._data_cloud.bucket.list_blobs()\
            if blob.name != "giotto-deep-big.png"])
        if self.name in existing_datasets:
            raise ValueError(f"Dataset {self.name} already exists in the cloud.\n Available datasets are: {existing_datasets}.")
        if path_metadata is None:
            path_metadata = self.path_metadata
        self._upload_metadata(path_metadata)
        self._upload_data(path_data)
        self._upload_label(path_label)
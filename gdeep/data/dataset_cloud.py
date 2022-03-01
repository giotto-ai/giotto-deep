from gdeep.data import _DataCloud
from typing import Union
from os import remove
from os.path import join, split, exists
import json

class DatasetCloud():
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
        # List of existing datasets in the cloud.
        existing_datasets = [blob.name.split('/')[0] for blob in self._data_cloud.bucket.list_blobs()]
        if self.name + '/' in existing_datasets:
            raise ValueError(f"Dataset {self.name} already exists in the cloud.")
        
    def __del__(self):
        if not self.path_metadata == None:
            remove(self.path_metadata)
            
    def download(self):
        # List of existing datasets in the cloud.
        existing_datasets = [blob.name.split('/')[0] for blob in self._data_cloud.bucket.list_blobs()]
        if not self.name in existing_datasets:
            raise ValueError(f"Dataset {self.name} does not exists in the cloud.")
        if exists(join(self.download_directory, self.name)):
            print(f"Dataset {self.name} already exists in the download directory.")
        self._data_cloud.download_folder(self.name + '/')

    @staticmethod
    def _get_filetype(path: str) -> str:
        return path.split('.')[-1]
    
    def _upload_data(self,
                path: str,) -> None:
        filetype = DatasetCloud._get_filetype(path)
        assert filetype in ('pt', 'npy'), "File type not supported."
        self._data_cloud.upload_file(path, self.metadata['name'] + '/data.' + filetype)  # type: ignore
    
    
    def _upload_label(self,
                 path: str,) -> None:
        filetype = DatasetCloud._get_filetype(path)
        assert filetype in ('pt', 'npy'), "File type not supported."
        self._data_cloud.upload_file(path, self.metadata['name'] + '/labels.' + filetype)  # type: ignore
    
    def _upload_metadata(self,
                        path: Union[str, None]=None):
        if self.metadata == None:
            raise Exception("No metadata to upload. Please create metadata using create_metadata.")  #NOSONAR
        self._data_cloud.upload_file(path, self.metadata['name'] + '/' + 'metadata.json')  # type: ignore

    def add_metadata(self,
                     size_dataset: int,
                     num_labels: int,
                     data_type: str = "tabular",
                     name: Union[None, str]=None,
                     data_format: Union[None, str]=None,
                     ):
        if name is None:
            name = self.name
        if data_format is None:
            data_format = "pytorch_tensor"
        self.path_metadata = "tmp_metadata.json"  # type: ignore
        self.metadata = {'name': name,
                    'size': size_dataset,
                    'num_labels': num_labels,
                    'data_type': data_type,
                    'data_format': data_format}
        with open(self.path_metadata, "w") as f:  # type: ignore
            json.dump(self.metadata, f, sort_keys=True, indent=4)
        
    def upload(self,
                       path_data: str,
                       path_label: str,
                       path_metadata: Union[str, None] = None,
                       ):
        if path_metadata is None:
            path_metadata = self.path_metadata
        self._upload_metadata(path_metadata)
        self._upload_data(path_data)
        self._upload_label(path_label)
from typing import List, Union
from os.path import join

class DatasetDownloader():
    """Class to download a dataset from a Google Cloud Storage bucket via the public URL.
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
        
    def download(self):
        """Download a dataset from the DatasetCloud.

        Raises:
            ValueError: Dataset does not exits in cloud.
        """
        pass
        
    def get_existing_datasets(self) -> List[str]:
        """Returns the list of existing datasets in the bucket.

        Returns:
            List[str]: List of datasets
        """
        pass

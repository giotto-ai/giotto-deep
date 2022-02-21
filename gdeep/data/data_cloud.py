from google.cloud import storage  # type: ignore
import os
from typing import Union

class DataCloud():
    def __init__(
            self,
            bucket_name: str ="adversarial_attack",
            download_directory: str = os.path.join('examples', 'data', 'DataCloud')
            ):
        """Download handle for Google Cloud Storage buckets.

        Args:
            bucket_name (str, optional): Name of the Google Cloud Storage bucket.
                Defaults to "adversarial_attack".
            download_directory (str, optional): Directory of the downloaded files.
                Defaults to os.path.join('examples', 'data', 'DataCloud').
        """
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Set up download path
        self.download_directory = download_directory
        
        # Create a new directory because it does not exist 
        if not os.path.exists(self.download_directory):
              os.makedirs(self.download_directory)

        
        
    def download(self,
                 source_blob_name: str,
                 destination_file_name: Union[str, None] = None):
        """Download a blob from Google Cloud Storage bucket.

        Args:
            source_blob_name (str): Name of the blob to download.
            destination_file_name (str): Name of the destination file to download.
        """
        if destination_file_name is None:
            destination_file_name = source_blob_name
        blob = self.bucket.blob(source_blob_name)
        blob.download_to_filename(os.path.join(self.download_directory, destination_file_name))
        
    def upload(self,
               source_file_name: str,
               destination_blob_name: str,
               upload_directory: Union[str, None] = None):
        """Upload a local file to Google Cloud Storage bucket.

        Args:
            source_file_name (str): Filename of the local file to upload. This
                has to be in the directory ´upload_directory´.
            source_blob_name (str): _description_
            upload_directory (str): Directory where the local file should be. Defaults to "".
        """
        if upload_directory is None:
            upload_directory = ""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(os.path.join(upload_directory, source_file_name))
        
    def delete_blob(self,
                    blob_name: str):
        blob = self.bucket.blob(blob_name)
        blob.delete()
        
        
        


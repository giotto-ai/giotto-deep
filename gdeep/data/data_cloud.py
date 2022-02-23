from google.cloud import storage  # type: ignore
from os.path import isfile, join, isdir, exists
from os import listdir, makedirs
import sys
import glob
from typing import Union
import logging

from sympy import true

class DataCloud():
    def __init__(
            self,
            bucket_name: str ="adversarial_attack",
            download_directory: str = join('examples', 'data', 'DataCloud')
            ) -> None:
        """Download handle for Google Cloud Storage buckets.

        Args:
            bucket_name (str, optional): Name of the Google Cloud Storage bucket.
                Defaults to "adversarial_attack".
            download_directory (str, optional): Directory of the downloaded files.
                Defaults to join('examples', 'data', 'DataCloud').
        """
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Set up download path
        self.download_directory = download_directory
        
        # Create a new directory because it does not exist 
        if not exists(self.download_directory):
              makedirs(self.download_directory)

        
        
    def download_file(self,
                 blob_name: str) -> None:
        """Download a blob from Google Cloud Storage bucket.

        Args:
            source_blob_name (str): Name of the blob to download.
        """
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(join(self.download_directory, blob_name))
        
    def download_folder(self,
                        blob_name: str) -> None:
        """Download a folder from Google Cloud Storage bucket.
        
        Warning: This function does not download empty subdirectories.

        Args:
            blob_name (str): Name of the blob folder to download.
        """
        # Get list of files in the blob
        blobs = self.bucket.list_blobs(prefix=blob_name)
        for blob in blobs:
            # Do not download subdirectories
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            if not exists(directory):
                makedirs(join(self.download_directory, directory), exist_ok=True)
            logging.getLogger().info("Downloading blob %s", blob.name)
            
            local_path =  blob.name.replace("/", "\\") if sys.platform == 'win32' else blob.name
            
            blob.download_to_filename(join(self.download_directory,local_path))
        
    def upload_file(self,
               source_file_name: str,
               target_blob_name: Union[str, None] = None) -> None:
        """Upload a local file to Google Cloud Storage bucket.

        Args:
            source_file_name (str): Filename of the local file to upload.
        """
        if target_blob_name is None:
            target_blob_name = source_file_name
        blob = self.bucket.blob(target_blob_name)
        if blob.exists():
            raise RuntimeError(f"Blob {source_file_name} already exists.")
        blob.upload_from_filename(source_file_name)
    
    def upload_folder(self,
                      source_folder: str
                      ) -> None:
        """Upload a local folder to Google Cloud Storage bucket recursively.

        Args:
            source_folder (str): Folder to upload.
            target_folder (Union[str, None], optional): Folder. Defaults to None.
        """
        
        assert isdir(source_folder)

        files_and_folders = [f for f in listdir(source_folder)]
        for f in files_and_folders:
            if(isfile(join(source_folder, f))):
                logging.getLogger().info("Create Blob %s", source_folder.replace("\\", "/"))
                logging.getLogger().info("upload file %s", join(source_folder, f))
                blob = self.bucket.blob(join(source_folder, f).replace("\\", "/"))
                blob.upload_from_filename(join(source_folder, f))
            else:
                self.upload_folder(join(source_folder, f))
        
    
    def delete_blob(self,
                    blob_name: str):
        """Deletes a single Blob from Google Cloud Storage

        Args:
            blob_name (str): The name of the Blob to delete
        """
        blob = self.bucket.blob(blob_name)
        blob.delete()
        
    def delete_blobs(self,
                     blobs_name: str) -> None:
        """Deletes a Blob and all its children from Google Cloud Storage.

        Args:
            blobs_name (str): Name of the parent Blob to delete.
        """
        blobs = self.bucket.list_blobs(prefix=blobs_name)
        for blob in blobs:
            blob.delete()





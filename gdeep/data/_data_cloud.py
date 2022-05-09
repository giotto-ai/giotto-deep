import logging
import os
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists, getsize
import requests  # type: ignore
import sys
from typing import Union, List
import time

import google  # type: ignore
from google.cloud import storage  # type: ignore
from google.oauth2 import service_account  # type: ignore
import wget  # type: ignore

from gdeep.utility import DEFAULT_DOWNLOAD_DIR, DATASET_BUCKET_NAME
from gdeep.utility.utils import get_checksum


def _check_public_access(use_public_access: bool):
    """Check if the public access is enabled."""
    def wrap(function):
        def wrapper_function(*args, **kwargs):
            if(use_public_access):
                raise ValueError("DataCloud object has public access only!")
            return function(*args, **kwargs)
        return wrapper_function
    return wrap

class _DataCloud():
    """Download handle for Google Cloud Storage buckets.

    Args:
        bucket_name (str, optional):
            Name of the Google Cloud Storage bucket.
            Defaults to "adversarial_attack".
        download_directory (str, optional):
            Directory of the downloaded files.
            Defaults to join('examples', 'data', 'DataCloud').
        use_public_access: (bool, optional):
            Whether or not to use public api access.
            Defaults to True.
        path_credentials (str, optional):
            Path to the credentials file.
            Only used if public_access is False and credentials are not
            provided. Defaults to None.
            
        Raises:
            ValueError: If the bucket does not exist.
            
        Returns:
            None
    """
    def __init__(
            self,
            bucket_name: str = DATASET_BUCKET_NAME,
            download_directory: str = DEFAULT_DOWNLOAD_DIR,
            use_public_access: bool = True,
            path_to_credentials: Union[str, None] = None,
            ) -> None:
        self.bucket_name = bucket_name
        self.use_public_access = use_public_access
        if not self.use_public_access:
            # Get storage client
            if path_to_credentials is None:
                self.storage_client = storage.Client()
            else:
                credentials = service_account.Credentials.\
                    from_service_account_file(path_to_credentials)
                self.storage_client = storage.Client(credentials=credentials)
            self.bucket = self.storage_client.bucket(self.bucket_name)
        else:
            self.public_url = ("https://storage.googleapis.com/" + 
                               bucket_name + "/")
        
        # Set up download path
        self.download_directory = download_directory
        
        # Create a new directory because it does not exist 
        if not exists(self.download_directory) \
            and self.download_directory != "":
              makedirs(self.download_directory)

    def list_blobs(self) -> List[str]:
        """List all blobs in the bucket.
        
        Returns:
            List[str]:
                List of blobs in the bucket.
        """
        # Assert that the bucket does not use public access
        if self.use_public_access:
            raise ValueError("DataCloud object can only list blobs"
                             "when using private access!")
        blobs = self.bucket.list_blobs()
        return [blob.name for blob in blobs]
    
    def blob_exists(self,
                    blob_name: str) -> bool:
        """Check if a Blob exists in the bucket.

        Args:
            blob_name (str):
                Name of the Blob to check.
                
            
        Returns:
            bool:
                True if the Blob exists, False otherwise.
        """
        if self.use_public_access:
            url = self.public_url + blob_name
            response = requests.head(url)
            return response.status_code == 200
        else:
            blob = self.bucket.blob(blob_name)
            return blob.exists() 
        
    def download_file(self,
                 blob_name: str,
                 download_directory: Union[str, None]=None) -> None:
        """Download a blob from Google Cloud Storage bucket.

        Args:
            source_blob_name (str):
                Name of the blob to download. The name is relative to the
                root of the bucket.
            download_directory (str, optional):
                Directory to download the blob to.
        
        Raises:
            ValueError:
                If the blob does not exist.
            
        Returns:
            None
        """
        if download_directory is None:
            download_directory = self.download_directory
        if self.use_public_access:
            url = self.public_url + blob_name
        # Check if blob exists
        if not self.blob_exists(blob_name):
            raise google.api_core.exceptions.NotFound(
                "Blob {} does not exist!".format(blob_name)
            )
        
        # If the file exists, compare checksums
        if isfile(join(download_directory, blob_name)):
            # Get remote md5 checksum from url in base64 format.
            if self.use_public_access:
                response = requests.get(url, stream=True)
                response.raw.decode_content = True
                # Get remote md5 checksum from url in base64 format if it exists.
                checksum_remote = response.headers.get("Content-MD5")
            else:
                blob = self.bucket.blob(blob_name)
                checksum_remote = blob.md5_hash
                
            checksum_local = get_checksum(
                join(download_directory, blob_name),
                encoding="base64"
                )
            if checksum_remote is not None:
                if checksum_remote != checksum_local:
                    # Ask user if they want to download the file
                    answer = input(
                        f"File {join(download_directory, blob_name)} already" +
                        "exists and checksums don't match! " +
                        "Do you want to overwrite it? [y/N]")
                    if answer.lower() not in  ["y", "yes"]:
                        return
                else: 
                    print(f"File {join(download_directory, blob_name)} " +
                          "already exists and checksums match! " +
                                "Skipping download.")
                    return
            else:
                print(f"File {join(download_directory, blob_name)} already" +
                      "exists and remote checksum is " +
                            "None! Downloading anyway.")
        # Download file
        print("Downloading file {} to {}".format(
            blob_name, download_directory))
        if self.use_public_access:
            wget.download(url, join(download_directory, blob_name))
        else:
            self.bucket.blob(blob_name)\
                .download_to_filename(join(download_directory, blob_name),
                                      checksum="md5")
        
    def download_folder(self,
                        blob_name: str) -> None:
        """Download a folder from Google Cloud Storage bucket.
        
        Warning: This function does not download empty subdirectories.

        Args:
            blob_name (str):
                Name of the blob folder to download. The name is relative to
                the root of the bucket.
        
        Raises:
            RuntimeError:
                If the folder does not exist.
        
        Returns:
            None
        """
        assert not self.use_public_access, ("Downloading folders is not"
                                            "is not supported with public"
                                            "access!")
        # Get list of files in the blob
        blobs = self.bucket.list_blobs(prefix=blob_name)
        for blob in blobs:
            # Do not download subdirectories
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            if not exists(directory):
                makedirs(join(self.download_directory, directory),
                         exist_ok=True)
            logging.getLogger().info("Downloading blob %s", blob.name)
            
            local_path =  blob.name.replace("/", "\\") \
                if sys.platform == 'win32' else blob.name
            
            blob.download_to_filename(join(self.download_directory,local_path),
                                      checksum="md5")
        
    def upload_file(self,
               source_file_name: str,
               target_blob_name: Union[str, None] = None,
               make_public: bool = False,
               overwrite: bool = False) -> None:
        """Upload a local file to Google Cloud Storage bucket.

        Args:
            source_file_name (str):
                Filename of the local file to upload.
            target_blob_name (Union[str, None], optional):
                Name of the target blob relative to the root of the bucket.
                If None, the filename will be used.
                Defaults to None.
            make_public (bool, optional):
                Whether or not to make the uploaded
                file public. Defaults to False.
                overwrite (bool, optional):
            Whether or not to overwrite the target
                Blob. Defaults to False.
            
        Raises:
            RuntimeError: If the target Blob already exists.
            
        Returns:
            None
        """
        if target_blob_name is None:
            target_blob_name = os.path.basename(source_file_name)
        blob = self.bucket.blob(target_blob_name)
        if blob.exists() and not overwrite:
            raise RuntimeError(f"Blob {target_blob_name} already exists.")
        logging.getLogger().info("upload file %s", source_file_name)
        # Check if source_file_name is bigger than 5GB
        if isfile(source_file_name) and\
            getsize(source_file_name) > 5000000000:
            raise ValueError("File is bigger than 5GB")
        
        # Compute MD5 checksum of the file and add it to the metadata of
        # the blob
        blob.md5_hash = get_checksum(source_file_name, encoding="base64")
        
        blob.upload_from_filename(source_file_name, checksum="md5")
        if make_public:
            blob.make_public()
    
    def upload_folder(self,
                      source_folder: str,
                      target_folder: str = None,
                      make_public: bool = False,
                      ) -> None:
        """Upload a local folder with all it's subolders to Google 
        Cloud Storage bucket.

        Args:
            source_folder (str):
                Folder to upload.
            target_folder (Union[str, None], optional):
                Name of the target folder relative to the root of the bucket.
                If None, the root of the bucket will be used.
                Defaults to None.
            make_public (bool, optional):
                Whether or not to make the uploaded
                file public. Defaults to False.
        
        Raises:
            ValueError:
                If the source folder is not a directory.
            
        Returns:
            None
        """
        if not isdir(source_folder):
            raise ValueError("Source folder is not a directory.")
        
        if target_folder is None:
            target_folder = ""
        
        # List of all files in the source folder
        files = [join(source_folder, f) for f in listdir(source_folder)
                 if isfile(join(source_folder, f))]
        print(files)
        # Upload all files in the source folder
        for file in files:
            file_name = os.path.basename(file)
            if(target_folder == ""):
                self.upload_file(join(source_folder, file_name),
                                target_blob_name=file_name,
                                make_public=make_public)
            else:
                self.upload_file(join(source_folder, file_name),
                                target_blob_name=target_folder + '/' + file_name,
                                make_public=make_public)
        
        # List of all subfolders in the source folder
        subfolders = [join(source_folder, f) for f in listdir(source_folder)
                        if isdir(join(source_folder, f))]
        # Upload all subfolders in the source folder recursively
        for subfolder in subfolders:
            relative_subfolder = os.path.relpath(subfolder, source_folder)
            if target_folder == "":
                self.upload_folder(join(source_folder, relative_subfolder),
                                target_folder=relative_subfolder,
                                make_public=make_public)
            else:
                self.upload_folder(join(source_folder, relative_subfolder),
                                target_folder=target_folder + '/' +\
                                    relative_subfolder,
                                make_public=make_public)

        
    
    def delete_blob(self,
                    blob_name: str) -> None:
        """Deletes a single Blob from Google Cloud Storage

        Args:
            blob_name (str):
                The name of the Blob to delete
                
        Raises:
            RuntimeError: If the Blob does not exist.
        
        Returns:
            None
        """
        blob = self.bucket.blob(blob_name)
        blob.delete()
        
    def delete_blobs(self,
                     blobs_name: str) -> None:
        """Deletes a Blob and all its children from Google Cloud Storage.

        Args:
            blobs_name (str):
                Name of the parent Blob to delete.
                
        Raises:
            ValueError: 
                If the Blob does not exist.
            
        Returns:
            None
        """
        blobs = self.bucket.list_blobs(prefix=blobs_name)
        for blob in blobs:
            blob.delete()





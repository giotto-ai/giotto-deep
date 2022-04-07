import logging
from os.path import isfile, join, isdir, exists, getsize
from os import listdir, makedirs
import sys
from typing import Union, List

from google.cloud import storage  # type: ignore
from google.oauth2 import service_account  # type: ignore

from gdeep.utility.utils import DEFAULT_DOWNLOAD_DIR, DATASET_BUCKET_NAME

LOGGER = logging.getLogger(__name__)


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
        if path_to_credentials is None:
            self.storage_client = storage.Client()
        else:
            credentials = service_account.Credentials.from_service_account_file(
                path_to_credentials)
            self.storage_client = storage.Client(credentials=credentials)
        
        print(self.bucket_name)
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Set up download path
        self.download_directory = download_directory
        
        # Create a new directory because it does not exist 
        if not exists(self.download_directory):
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
        
        
    def download_file(self,
                 blob_name: str) -> None:
        """Download a blob from Google Cloud Storage bucket.

        Args:
            source_blob_name (str):
                Name of the blob to download.
        
        Raises:
            ValueError:
                If the blob does not exist.
            
        Returns:
            None
        """
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(join(self.download_directory, blob_name))
        
    def download_folder(self,
                        blob_name: str) -> None:
        """Download a folder from Google Cloud Storage bucket.
        
        Warning: This function does not download empty subdirectories.

        Args:
            blob_name (str):
                Name of the blob folder to download.
        
        Raises:
            RuntimeError:
                If the folder does not exist.
        
        Returns:
            None
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
                makedirs(join(self.download_directory, directory),
                         exist_ok=True)
            logging.getLogger().info("Downloading blob %s", blob.name)
            
            local_path =  blob.name.replace("/", "\\") \
                if sys.platform == 'win32' else blob.name
            
            blob.download_to_filename(join(self.download_directory,local_path))
        
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
                Name of the target Blob. Defaults to None.
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
            target_blob_name = source_file_name
        blob = self.bucket.blob(target_blob_name)
        if blob.exists() and not overwrite:
            raise RuntimeError(f"Blob {target_blob_name} already exists.")
        logging.getLogger().info("upload file %s", source_file_name)
        # Check if source_file_name is bigger than 5GB
        if isfile(source_file_name) and\
            getsize(source_file_name) > 5000000000:
            raise ValueError("File is bigger than 5GB")
                     
        
        blob.upload_from_filename(source_file_name)
        if make_public:
            blob.make_public()
    
    def upload_folder(self,
                      source_folder: str,
                      make_public: bool = False,
                      ) -> None:
        """Upload a local folder to Google Cloud Storage bucket recursively.

        Args:
            source_folder (str):
                Folder to upload.
            target_folder (Union[str, None], optional):
                Folder. Defaults to None.
            make_public (bool, optional):
                Whether or not to make the uploaded
                file public. Defaults to False.
        
        Raises:
            ValueError:
                If the source folder is not a directory.
            
        Returns:
            None
        """
        
        assert isdir(source_folder)

        files_and_folders = [f for f in listdir(source_folder)]
        for f in files_and_folders:
            if(isfile(join(source_folder, f))):
                logging.getLogger()\
                    .info("Create Blob %s", source_folder.replace("\\", "/"))
                logging.getLogger()\
                    .info("upload file %s", join(source_folder, f))
                blob = self.bucket\
                    .blob(join(source_folder, f).replace("\\", "/"))
                blob.upload_from_filename(join(source_folder, f))
                if make_public:
                    blob.make_public()
            else:
                self.upload_folder(join(source_folder, f))
        
    
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





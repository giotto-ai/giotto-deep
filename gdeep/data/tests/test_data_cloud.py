from ..data_cloud import DataCloud
import os
import hashlib
import google  # type: ignore
import logging
import random

LOGGER = logging.getLogger(__name__)
import pytest

def credentials_error_logging(func):
    def inner():
        try:
            func()
        except google.auth.exceptions.DefaultCredentialsError:
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
def test_download():
    """Test download of sample data from bucket
    """
    data_cloud = DataCloud()
    file_name = "giotto-deep-big.png"
    data_cloud.download(file_name)
    
    # check if correct extension is raised when trying to download non-existing file
    with pytest.raises(google.api_core.exceptions.NotFound):
        non_existing_file_name: str = "giotto-deep-bigs.png"
        data_cloud.download(non_existing_file_name)
    
    # check if downloaded file exists
    file_path = os.path.join(data_cloud.download_directory, file_name)
    assert os.path.exists(file_path)

    # check if downloaded file is correct
    assert "d4b12b2dc2bc199831ba803431184fcb" == \
        hashlib.md5(file_as_bytes(open(file_path, 'rb'))).hexdigest()
        
@credentials_error_logging
def test_upload():
    """Test upload of sample file to bucket.
    """
    data_cloud = DataCloud()
    
    # create temporary file to upload to bucket
    sample_file_name = "tmp.txt"
    sample_text = "Create a new tmp file!" + str(random.randint(0, 1_000))
    
    if os.path.exists(sample_file_name):
        os.remove(sample_file_name)
    with open(sample_file_name, 'w') as f:
        f.write(sample_text)
    
    assert os.path.exists(sample_file_name)
    
    # upload sample file to bucket
    data_cloud.upload_file(sample_file_name)
    
    # delete local temporary file
    os.remove(sample_file_name)
    
    data_cloud.download(sample_file_name)
    
    data_cloud.delete_blob(sample_file_name)
    
    # check if downloaded file exists
    file_path = os.path.join(data_cloud.download_directory, sample_file_name)
    assert os.path.exists(file_path)
    
    with open(file_path, 'r') as f:
        assert f.read() == sample_text
        
# def test_upload_folder():
#     data_cloud = DataCloud()
    
#     # create temporary folder and temporary file to upload to bucket
#     sample_dir = 'tmp'
#     sample_file_name = os.path.join(sample_dir, "tmp.txt")
#     sample_text = "Create a new tmp file!" + str(random.randint(0, 1_000))
    
#     if not os.path.exists(sample_dir):
#         os.makedirs(sample_dir)
    
#     if os.path.exists(sample_file_name):
#         os.remove(sample_file_name)
#     with open(sample_file_name, 'w') as f:
#         f.write(sample_text)
    
#     assert os.path.exists(sample_file_name)
    
#     # upload sample file to bucket
#     data_cloud.upload_file(sample_dir,
#                       sample_dir)
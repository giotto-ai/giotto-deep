from gdeep.data import _DataCloud
from os.path import join, exists
from os import remove, makedirs, environ
import hashlib
import google  # type: ignore
import logging
import random
from shutil import rmtree
import pytest
from google.auth.exceptions import DefaultCredentialsError  # type: ignore

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

if "GOOGLE_APPLICATION_CREDENTIALS" in dict(environ):
    @credentials_error_logging
    def test_download():
        """Test download of sample data from bucket
        """
        data_cloud = _DataCloud()
        file_name = "giotto-deep-big.png"
        data_cloud.download_file(file_name)
        
        # check if correct extension is raised when trying to download non-existing file
        non_existing_file_name: str = "giotto-deep-bigs.png"
        with pytest.raises(google.api_core.exceptions.NotFound):
            data_cloud.download_file(non_existing_file_name)
        remove(join(data_cloud.download_directory, non_existing_file_name))
            
        
        # check if downloaded file exists
        file_path = join(data_cloud.download_directory, file_name)
        assert exists(file_path)

        # check if downloaded file is correct
        assert "d4b12b2dc2bc199831ba803431184fcb" == \
            hashlib.md5(file_as_bytes(open(file_path, 'rb'))).hexdigest()
            
        remove(join(data_cloud.download_directory, file_name))
            
    @credentials_error_logging
    def test_upload():
        """Test upload of sample file to bucket.
        """
        data_cloud = _DataCloud()
        
        # create temporary file to upload to bucket
        sample_file_name = "tmp.txt"
        sample_text = "Create a new tmp file!" + str(random.randint(0, 1_000))
        
        if exists(sample_file_name):
            remove(sample_file_name)
        with open(sample_file_name, 'w') as f:
            f.write(sample_text)
        
        assert exists(sample_file_name)
        
        # upload sample file to bucket
        data_cloud.upload_file(sample_file_name)
        
        # check if uploading to an already existing file raises exception
        with pytest.raises(RuntimeError):
            data_cloud.upload_file(sample_file_name)
            
        # delete local temporary file
        remove(sample_file_name)
        
        data_cloud.download_file(sample_file_name)
        
        data_cloud.delete_blob(sample_file_name)
        
        # check if downloaded file exists
        file_path = join(data_cloud.download_directory, sample_file_name)
        assert exists(file_path)
        
        with open(file_path, 'r') as f:
            assert f.read() == sample_text
        
        remove(file_path)

    def test_upload_folder():
        """Test the upload of a folder to bucket and download the folder.
        """
        data_cloud = _DataCloud()
        
        # create temporary folder structure and temporary file to upload to bucket
        # tmp: tmp.txt
        # |- sub_tmp_1: tmp1.txt
        # |- sub_tmp_2: tmp2.txt
        #     |- sub_tmp_2_2: tmp2_2.txt
        if exists('tmp'):
            rmtree('tmp')
        
        tmp_files = []
        
        sample_dir = 'tmp'
        makedirs(sample_dir)
        tmp_files.append(join(sample_dir, "tmp.txt"))
        
        sub_1_sample_dir = join(sample_dir, 'sub_tmp_1')
        makedirs(sub_1_sample_dir)
        tmp_files.append(join(sub_1_sample_dir, "tmp1.txt"))
        
        sub_2_sample_dir = join(sample_dir, 'sub_tmp_2')
        makedirs(sub_2_sample_dir)
        tmp_files.append(join(sub_2_sample_dir, "tmp2.txt"))
        
        sub_2_2_sample_dir = join(sub_2_sample_dir, 'sub_tmp_2_2')
        makedirs(sub_2_2_sample_dir)
        tmp_files.append(join(sub_2_2_sample_dir, "tmp2_2.txt"))

        
        sample_texts = {}
        
        for file in tmp_files:
            if exists(file):
                remove(file)
            sample_text = "Create a new tmp file! " + str(random.randint(0, 1_000))
            sample_texts[file] = sample_text
            with open(file, 'w') as f:
                f.write(sample_text)
            assert exists(file)
        
        # upload sample file to bucket
        data_cloud.upload_folder(sample_dir)
        
        # delete local tmp folder
        rmtree(sample_dir)
        
        # download folder to local
        data_cloud.download_folder(sample_dir)
        
        # delete folder in bucket
        data_cloud.delete_blobs(sample_dir)
        
        # check if downloaded folder is correct
        
        for file in sample_texts.keys():
            with open(join(data_cloud.download_directory, file), 'r') as f:
                assert f.read() == sample_texts[file]
        
        # delete local tmp folder
        rmtree(join(data_cloud.download_directory, sample_dir))
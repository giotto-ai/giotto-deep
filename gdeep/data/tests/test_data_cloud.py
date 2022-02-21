from ..data_cloud import DataCloud
import os
import hashlib
import google
import logging

LOGGER = logging.getLogger(__name__)
import pytest

def test_download():
    """Test download of sample data from bucket
    """
    try:
        data_cloud = DataCloud()
        file_name = "giotto-deep-big.png"
        data_cloud.download(file_name,
                            file_name)
        
        # check if correct extension is raised when trying to download non-existing file
        with pytest.raises(google.api_core.exceptions.NotFound):
            non_existing_file_name: str = "giotto-deep-bigs.png"
            data_cloud.download(non_existing_file_name,
                                non_existing_file_name)
        
        # check if downloaded file exists
        file_path = os.path.join(data_cloud.download_directory, file_name)
        assert os.path.exists(file_path)
        
        def file_as_bytes(file):
            with file:
                return file.read()

        # check if downloaded file is correct
        assert "d4b12b2dc2bc199831ba803431184fcb" == hashlib.md5(file_as_bytes(open(file_path, 'rb'))).hexdigest()
    except google.auth.exceptions.DefaultCredentialsError:
        LOGGER.warning("GCP credentials failed.")
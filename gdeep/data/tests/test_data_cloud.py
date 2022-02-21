from ..data_cloud import DataCloud
import os
import hashlib

def test_download():
    """Test download of sample data from bucket
    """
    data_cloud = DataCloud()
    file_name = "giotto-deep-big.png"
    data_cloud.download(file_name,
                        file_name)
    
    # check if downloaded file exists
    file_path = os.path.join(data_cloud.download_directory, file_name)
    assert os.path.exists(file_path)
    
    def file_as_bytes(file):
        with file:
            return file.read()

    # check if downloaded file is correct
    assert "d4b12b2dc2bc199831ba803431184fcb" == hashlib.md5(file_as_bytes(open(file_path, 'rb'))).hexdigest()
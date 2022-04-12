# %%
from gdeep.utility.utils import get_checksum, autoreload_if_notebook
autoreload_if_notebook()
# %%
from gdeep.data import _DataCloud
from gdeep.utility.constants import DATASET_BUCKET_NAME


# file C:\Users\Raphael\Downloads\1920x1080.jpg
file_path = "C:\\Users\\Raphael\\Downloads\\1920x1080.jpg"
# %%
dc = _DataCloud(DATASET_BUCKET_NAME,
                use_public_access=False,)

target_name = "sample_image.jpg"

get_checksum(file_path)

dc.upload_file(file_path,
               target_name,
               make_public=True)
# %%
target_name = "sample_image.jpg"
dc2 = _DataCloud(DATASET_BUCKET_NAME,
                use_public_access=False,)

dc2.download_file(target_name)

# %%
from gdeep.data import _DataCloud
data_cloud = _DataCloud(use_public_access=False)
file_name = "giotto-deep-big.png"
data_cloud.download_file(file_name)
# %%

import os

# Define the root directory of the project which is parent of the parent of 
# the current directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir))

# Define the default dataset download directory
DEFAULT_DOWNLOAD_DIR = os.path.join(ROOT_DIR, "examples", "data",
                                    "DatasetCloud")

# Define the default dataset bucket on Google Cloud Storage where the datasets
# are stored
DATASET_BUCKET_NAME = "adversarial_attack"
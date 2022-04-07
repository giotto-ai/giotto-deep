from gdeep.search import PersformerHyperparameterSearch

import os
from os import remove, listdir
from os.path import join
from shutil import rmtree

from gdeep.utility.utils import ROOT_DIR

def clean_up_files(fun):
    def wrapper():
        # Get all subdirectories of the root directory
        directories = [d for d in listdir(ROOT_DIR) if os.path.isdir(join(ROOT_DIR, d))]

        # Get all files in the root directory
        files = [f for f in listdir(ROOT_DIR) if os.path.isfile(join(ROOT_DIR, f))]


        fun()
        # Get all subdirectories of the root directory
        new_directories = [d for d in listdir(ROOT_DIR) if os.path.isdir(join(ROOT_DIR, d))]

        # Get all files in the root directory
        new_files = [f for f in listdir(ROOT_DIR) if os.path.isfile(join(ROOT_DIR, f))]

        # Delete all newly created directories
        for directory in new_directories:
            if directory not in directories:
                rmtree(join(ROOT_DIR, directory))

        # Delete all newly created files
        for file in new_files:
            if file not in files:
                remove(join(ROOT_DIR, file))
    return wrapper

@clean_up_files
def test_start():
    """Test the start method of the PersformerHyperparameterSearch class."""
    dataset_name = 'SmallMutagDataset'
    download_directory = join('gdeep','search', 'tests', 'data')
    path_hpo_metadata = join('gdeep','search', 'tests', 'data','hpo_metadata.json')
    path_writer = None
    

    hpo = PersformerHyperparameterSearch(dataset_name, download_directory, path_hpo_metadata, path_writer)

    hpo.search()
    

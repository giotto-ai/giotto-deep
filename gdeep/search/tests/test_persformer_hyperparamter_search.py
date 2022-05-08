from gdeep.search import PersformerHyperparameterSearch

import os
from os import remove, listdir
from os.path import join
from shutil import rmtree

from gdeep.utility import ROOT_DIR

from gdeep.search import clean_up_files


@clean_up_files
def test_start():
    """Test the start method of the PersformerHyperparameterSearch class."""
    dataset_name = 'SmallMutagDataset'
    download_directory = join('gdeep','search', 'tests', 'data')
    path_hpo_metadata = join('gdeep','search', 'tests', 'data','hpo_metadata.json')
    path_writer = None
    

    hpo = PersformerHyperparameterSearch(dataset_name, download_directory, path_hpo_metadata, path_writer)

    hpo.search()
    

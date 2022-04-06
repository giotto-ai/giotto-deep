from gdeep.search import PersformerHyperparameterSearch

import os
from os import remove, listdir
from os.path import join
from shutil import rmtree

from gdeep.utility.utils import ROOT_DIR

def test_start():
    """Test the start method of the PersformerHyperparameterSearch class."""
    dataset_name = 'SmallMutagDataset'
    download_directory = join('gdeep','search', 'tests', 'data')
    path_hpo_metadata = join('gdeep','search', 'tests', 'data','hpo_metadata.json')
    path_writer = None

    hpo = PersformerHyperparameterSearch(dataset_name, download_directory, path_hpo_metadata, path_writer)

    hpo.search()
    
    # Delete the writer directory
    writer_directory = join(ROOT_DIR, 'runs')
    rmtree(writer_directory)
    
    # Delete all events* and 202* files in the ROOT_DIR
    for file in listdir(ROOT_DIR):
        if file.startswith('events') or file.startswith('202'):
            remove(join(ROOT_DIR, file))

    
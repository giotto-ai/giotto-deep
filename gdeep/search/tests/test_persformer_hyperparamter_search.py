import pytest

from gdeep.search import PersformerHyperparameterSearch
from os.path import join

def test_start():
    """Test the start method of the PersformerHyperparameterSearch class."""
    dataset_name = 'SmallMutagDataset'
    download_directory = join("examples","data", "DatasetCloud")
    path_hpo_metadata = join('gdeep','search', 'tests', 'data','hpo_metadata.json')
    path_writer = '.'

    hpo = PersformerHyperparameterSearch(dataset_name, download_directory, path_hpo_metadata, path_writer)

    hpo.search()
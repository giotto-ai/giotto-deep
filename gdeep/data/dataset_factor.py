import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from gdeep.data import CreateToriDataset
from gdeep.utility import DEFAULT_DOWNLOAD_DIR

# Define the dataset factory class for the tori dataset and torchvision datasets
# using the factory design pattern
# https://realpython.com/factory-method-python/


class DatasetFactory(object):
    """ DataLoader factory class for the tori dataset and torchvision datasets
    using the factory design pattern
    
    Example::
        # Create a dataset for the tori dataset
        dataset = get_dataset("Tori", name="DoubleTori", n_points=100)
        
        # Create the MNIST dataset
        dataset = get_dataset("Torchvision", name="MNIST")
    """
    def __init__(self):
        self._builders = {}
        
    def register_builder(self, key, builder):
        self._builders[key] = builder
        
    def build(self, key, **kwargs):
        builder = self._builders.get(key)
        if builder is None:
            raise ValueError(f"No builder registered for key {key}")
        return builder(**kwargs)


class TorchvisionDatasetBuilder(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def __call__(self, **kwargs):
        return datasets.__getattribute__(self.dataset_name)(
            root=DEFAULT_DOWNLOAD_DIR, **kwargs
        )

        
class ToriDatasetBuilder(object):
    def __init__(self, name):
        self.name = name
        
    def __call__(self, **kwargs):
        return CreateToriDataset(name=self.name, **kwargs)
        
        
        
def get_dataset(key, **kwargs):
    """ Get a dataset from the factory
     
    Args:
        key (str): The key of the dataset to get
        **kwargs: The keyword arguments to pass to the dataset builder
        
    Returns:
        torch.utils.data.Dataset: The dataset
    """
    factory = DatasetFactory()
    factory.register_builder("Torchvision", TorchvisionDatasetBuilder())
    factory.register_builder("Tori", TorchvisionDatasetBuilder())
    return factory.build(key, **kwargs)
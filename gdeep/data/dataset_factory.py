
from typing import Dict, Any



# Define the dataset factory class for the tori dataset and torchvision datasets
# using the factory design pattern
# https://realpython.com/factory-method-python/


class DatasetFactory(object):
    """ Dataset factory class for the tori dataset and torchvision datasets
    using the factory design pattern
    
    Examples::

        # Create a dataset for the tori dataset
        dataset = get_dataset("Tori", name="DoubleTori", n_points=100)
        
        # Create the MNIST dataset
        dataset = get_dataset("Torchvision", name="MNIST")

    """

    _builders: Dict[str, Any]

    def __init__(self):
        self._builders = {}
        
    def register_builder(self, key: str, builder:Any):
        """this method adds to the internal builders dictionary
        new dataloader builders
        """
        self._builders[key] = builder    # type: ignore
        
    def build(self, key:str, **kwargs) -> Any:  # type: ignore
        """This method returns the DataLoader builder
        corresponding to the input key.

        Args:
            key:
                the name of the dataset
        """
        builder = self._builders.get(key)    # type: ignore
        if builder is None:
            raise ValueError(f"No builder registered for key {key}")
        return builder(**kwargs)  # type: ignore


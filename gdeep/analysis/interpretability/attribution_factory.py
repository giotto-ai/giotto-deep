
from typing import Dict, Any

from captum import attr

# Define the dataset factory class for the tori dataset and torchvision datasets
# using the factory design pattern
# https://realpython.com/factory-method-python/


class AttributionFactory(object):
    """ Attribution factory class for the Captum integration.
    This factory will contain the attributions techniques of captum.
    """

    _builders: Dict[str, Any] = {}

    def __init__(self):
        pass
        
    def register_builder(self, key: str, builder: Any):
        """this method adds to the internal builders dictionary
        new dataloader builders
        """
        self._builders[key] = builder    # type: ignore
        
    def build(self, key: str, *args, **kwargs) -> Any:  # type: ignore
        """This method returns the DataLoader builder
        corresponding to the input key.

        Args:
            key:
                the name of the dataset
        """
        builder = self._builders.get(key)    # type: ignore
        if builder is None:
            raise ValueError(f"No builder registered for key {key}")
        return builder(*args, **kwargs)  # type: ignore


class AttributionBuilder(object):
    """Builder class for the torchvision dataset
        and all its variations"""

    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def __call__(self, *args, **kwargs):  # type: ignore
        return attr.__getattribute__(self.attr_name)(  # type: ignore
            *args, **kwargs  # type: ignore
        )


def get_attr(key: str, *args, **kwargs) -> Any:  # type: ignore
    """Get a dataset from the factory

    Args:
        key :
            The name of the dataset, corresponding
            to the key in the list of builders
        **kwargs:
            The keyword arguments to pass to the dataset builder

    Returns:
        captum.attr:
            The captum interpretability tool
    """
    factory = AttributionFactory()

    for attr_name in attr.__all__:
        factory.register_builder(attr_name, AttributionBuilder(attr_name))

    return factory.build(key, *args, **kwargs)  # type: ignore

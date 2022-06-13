from inspect import signature
from typing import Any, Callable, List, Tuple, Union

import torch
from numpy import ndarray

Tensor = torch.Tensor


def torch_transform(
    transform: Union[Callable[[Tensor], Tensor], Callable[[ndarray], ndarray]]
) -> Callable[[Tensor], Tensor]:
    """ Transforms a numpy array transform to a torch transform. 
    
    Args:
        transform: Either a callable that takes a numpy array and returns a
            numpy array or a callable that takes a tensor and returns a tensor.
        
    Returns:
        The torch transform.
    """
    if (
        get_parameter_types(transform)[0] is Tensor
        and get_return_type(transform) is Tensor
    ):
        return transform  # type: ignore
    elif (
        get_parameter_types(transform)[0] is ndarray
        and get_return_type(transform) is ndarray
    ):
        return lambda x: torch.tensor(transform(x.numpy()))
    else:
        raise ValueError(
            "Transform must be a function that takes a tensor or"
            "an array and returns a tensor or an array."
            "Please provide type annotations."
        )


def get_parameter_types(func: Callable) -> List[Any]:
    """ Returns a list of the types of the parameters of a function. 
    
    Args:
        func: The function to get the types of the parameters of.
    Returns:
        A list of the types of the parameters of the function.
    """
    return [t.annotation for t in signature(func).parameters.values()]


def get_return_type(func: Callable) -> Any:
    """ Returns the type of the return value of a function.
     
    Args:
        func: The function to get the type of the return value of.
        
    Returns:
        The type of the return value of the function.
    """
    return signature(func).return_annotation

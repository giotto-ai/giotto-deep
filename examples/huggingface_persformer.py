# %%
from inspect import signature
from typing import Callable, Union, get_origin
import typing_inspect  # type: ignore
from typeguard import check_type
import torch
from gdeep import utility

from gdeep.data import PersistenceDiagramFromGraphDataset
from gdeep.utility import autoreload_if_notebook
autoreload_if_notebook()
# %%
# graph_dataset_name = 'MUTAGs'
# persistence_dataset = PersistenceDiagramFromGraphDataset(
#     graph_dataset_name,
#     10.1,
# )
# %%
# def f(x: torch.Tensor) -> torch.Tensor:
#     return x.size()
# def fun_check(g: Callable[[str], str]) -> bool:
#     return g("Hallo") == "Hallo"
# fun_check(f)
# %%
@strict
def f(x: int) -> int:
    return x + 1
def g(x: str) -> str:
    return x + "a"
# %%
def fun_check(h: Union[Callable[[str], str],
                       Callable[[int], int]]) -> bool:
    if isinstance(h, 
            return h("Hallo") == "Hallo"
    else:
        return h(1) == 2
# %%
typing_inspect.get_origin(type(f))
# %%
typing_inspect.get_args(type(f))
# %%
from inspect import signature
# %%
signature(f)
# %%
def f(x:int, y:int) -> int:
    return x + y
# %%
from gdeep.utility._typing_utils import get_parameter_types

# %%
class A:
    def __call__(self, x) -> int:
        return x + 1
        
get_parameter_types(A())

# %%

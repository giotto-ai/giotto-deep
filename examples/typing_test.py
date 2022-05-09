from typing import NewType  
import torch
PositiveInt = NewType('PositiveInt', int)

def assert_positive(n: int) -> PositiveInt:
    """
    Assert that the argument is positive.
    """
    assert n >= 0, "The argument must be positive!"
    return PositiveInt(n)

x: PositiveInt = assert_positive(-5)

IntTensor = torch.Tensor[int]


# %%
# from typing import Iterable, Iterator

# def print(items: Iterable) -> None:
#     """
#     Print the specified items.
#     """
#     for item in items:
#         print(item)

# class MyClass(Iterable):
#     """
#     A class that can be iterated over.
#     """
#     def __init__(self) -> None:
#         """
#         Initialize the class.
#         """
#         print("MyClass.__init__()")
#         return self
    
#     def __iter__(self) -> Iterator[int]:
#         """
#         Return an iterator over the class.
#         """
#         yield 1
#         yield 2
#         yield 3

# %%
def print_int(a: int) -> None:
    """
    Print the specified items.
    """
    print(a)


class MyClass:
    def __init__(self):
        """
        Initialize the class.
        """
        print("MyClass.__init__()")
        
    def __repr__(self) -> str:
        return "MyClass.__repr__()"

    # A method that returns a reference to self
    def __call__(self) -> 'MyClass':
        return self

a = MyClass()
b = a()
# %%

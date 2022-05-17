# %%
from typing import Any



class C:
    def f(self):
        print("Hello from C")

class D:
    def __init__(self, c: C):
        self.b = 1
        self.c = c
    def g(self):
        print("Hello from D")
    
    # Forward the call of the function f to the object c
    def __getattr__(self, name: str) -> Any:
        if name == "g":
            return self.name
        return getattr(self.c, name)
        
c = C()
d = D(c)
d.f()
d.g()
print(d.c)
print(c)
d.b

# Output:
# Hello from C
# Hello from D
# %%
import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from gdeep.utility.constants import DEFAULT_DOWNLOAD_DIR


ds = MNIST(root=DEFAULT_DOWNLOAD_DIR, train=True, download=True)
# %%
ds_sub = Subset(ds, range(10))
# %%
ds[:3]
# %%

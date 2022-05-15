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
d.__getattr__("g")
# %%

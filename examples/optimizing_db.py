# %%
import torch

# %%
W = torch.rand((10, 5))  # N weights, at the optimality
W.requires_grad = True
a = torch.rand((5,))
a.requires_grad = True
loss = torch.trace(torch.sum(a @ W.T) + W @ W.T)

L = torch.autograd.grad(loss, W, retain_graph=True, create_graph=True)

torch.autograd.grad(L[0], a, grad_outputs=torch.ones((10, 5)))
# %%
x = torch.rand(2)
x.requires_grad = True
torch.autograd.grad(2*x, x,  grad_outputs=torch.ones(2))
# %%
W = torch.rand((10, 5))
W.requires_grad = True
a = torch.rand((5,))
a.requires_grad = True


def opt_cond(a, W):
    loss = torch.trace(torch.sum(a @ W.T) + W @ W.T)

    L = torch.autograd.grad(loss, W, create_graph=True)
    return L


y = torch.autograd.functional.jacobian(
    lambda a: opt_cond(a, W), a
    )[0].reshape(50, 5)
A = - torch.autograd.functional.jacobian(
    lambda W: opt_cond(a, W), W
    )[0].reshape(50, 50)
# %%


def opt_cond_circ(x, y):
    return x**2 + y**2 - 1


x = torch.sqrt(torch.tensor(0.5, requires_grad=True))
y = torch.sqrt(torch.tensor(0.5, requires_grad=True))
A = - torch.autograd.grad(opt_cond_circ(x, y), x)[0]
z = torch.autograd.grad(opt_cond_circ(x, y), y)[0]
print(z / A)

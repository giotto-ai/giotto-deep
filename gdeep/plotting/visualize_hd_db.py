import torch


class LowDimensionalPlane():
    """
    Simulates a low dimensional plane at origin x and
    directions v=(v[0], v[1],...). The class has the
    functionality to projection points in the high
    dimensional surrounding space of dimension
    (n_1, n_2, ...) to the plane with the basis v
    as well as embedding points in the
    plane of dimension (len(v)) with respect to
    the v basis into the high dimensional space.
    """

    def __init__(self, x, v):
        """
        Args:
            x (torch.tensor): origin of the plane of shape (n_1, n_2, ...)
            v (torch.Tensor): directions of the low dimensional plane
                v[0], v[1],... . v has shape (dim, n_1, n_2, ...).
        """
        self.shape = x.shape  # shape is (n_1, n_2, ...)
        self.x = x
        self.v = []
        self.dim = v.shape[0]  # dimension of the low dimensional plane
        assert self.shape == v.shape[1:], \
            f'tensor v of shape {v.shape} has to have\
            the shape ({(self.dim,)+ tuple(self.shape)}).'
        self.v = v

        def gram_schmidt(vv):
            """Produces an orthonormal basis for the span of vv

            Args:
                vv (torch.Tensor): vectors v[0], v[1], ...
            """
            def projection(u, v):
                return (v * u).sum() / (u * u).sum() * u

            nk = vv.size(0)
            uu = torch.zeros_like(vv, device=vv.device)
            uu[0] = vv[0].clone()
            for k in range(1, nk):
                vk = vv[k].clone()
                uk = 0
                for j in range(0, k):
                    uj = uu[j].clone()
                    uk = uk + projection(uj, vk)
                uu[k] = vk - uk
            for k in range(nk):
                uk = uu[k].clone()
                uu[k] = uk / uk.norm()
            return uu
        self.u = gram_schmidt(self.v.reshape((self.v.shape[0], -1))) \
            .reshape(self.v.shape)
        self.v_flat = self.v.reshape((self.v.shape[0], -1))
        self.u_flat = self.u.reshape((self.u.shape[0], -1))
        self.u2v = (self.v_flat @ self.u_flat.t()).inverse()

        rand_input = torch.rand((10, self.dim))
        rand_embed = self.embed_tensor(rand_input)
        rand_project = self.project_tensor(rand_embed)
        assert torch.allclose(rand_project, rand_input, atol=1e-6), \
            "error in gram schmidt algorithm"
        assert torch.allclose(self.embed_tensor(torch.eye(self.dim)),
                              self.v + self.x.unsqueeze(0), atol=1e-6), \
            "v is not correctely embedded!"

    def embed_tensor(self, y_proj):
        """ Embeds a tensor in the low dimensional plane to the high
        dimensional surrounding space

        Args:
            y_proj (torch.Tensor): tensor in the low dimensional plane of shape
            (batch_size, dim).

        Returns:
            torch.Tensor: tensor in surrounding space of shape
            (batch_size, n_1, n_2, ...).
        """
        assert y_proj.shape[-1] == self.dim,\
            f'tensor of shape {y_proj.shape} has to be of shape\
                (*, {self.dim})'

        return self.x.unsqueeze(0) \
            + torch.einsum('d...,bd->b...', self.v, y_proj)

    def project_tensor(self, y):
        """ Project tensor to low dimensional plane

        Args:
            y (torch.Tensor): tensor in surrounding space of shape
            (batch_size, n_1, n_2, ...).

        Returns:
            torch.Tensor: tensor in the low dimensional plane of
            shape (batch_size, dim).
        """
        assert y.shape[1:] == self.shape,\
            f'tensor of shape {y.shape} has to be of shape (*, {self.shape})'

        # <y-x, u[i]>/||u[i]|| for all i
        len_u = len(self.u.shape)
        tuple_len_u_shifted = tuple(range(2, len_u + 1))

        y_proj = ((y.unsqueeze(1) - self.x.unsqueeze(0).unsqueeze(0)) *
                  self.u.unsqueeze(0)) \
            .sum(tuple_len_u_shifted)
        return y_proj @ self.u2v

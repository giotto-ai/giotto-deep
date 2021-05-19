# from https://github.com/juho-lee/set_transformer/blob/master/max_regression_demo.ipynb  # noqa: E501

import torch
import torch.nn as nn
from gdeep.topology_layers.modules import SAB, PMA  # type: ignore


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x


class SmallSetTransformer(nn.Module):
    def __init__(self,
                 dim_in: int = 1,
                 dim_out: int = 64,
                 num_heads: int = 4,
                 out_features: int = 1):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_in, dim_out=dim_out, num_heads=num_heads),
            SAB(dim_in=dim_out, dim_out=dim_out, num_heads=num_heads),
        )
        self.dec = nn.Sequential(
            PMA(dim=dim_out, num_heads=num_heads, num_seeds=1),
            nn.Linear(in_features=dim_out, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): Batch tensor of shape
                [batch, sequence_length, dim_in]

        Returns:
            torch.Tensor: [description]
        """
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)

    def num_params(self) -> int:
        """Returns number of trainable parameters.

        Returns:
            int: Number of trainable parameters.
        """
        total_params = 0
        for parameter in self.parameters():
            total_params += parameter.nelement()
        return total_params

# def train(model):
#     model = model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = nn.L1Loss().cuda()
#     losses = []
#     for _ in range(500):
#         x, y = gen_data(batch_size=2 ** 10, max_length=10)
#         x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()
#         loss = criterion(model(x), y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
#     return losses
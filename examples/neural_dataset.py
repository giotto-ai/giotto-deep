# %%
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from gdeep.search.hpo import HyperParameterOptimization

from gdeep.trainer import Trainer
from gdeep.search import GiottoSummaryWriter

# %%
writer = GiottoSummaryWriter()

# %%
path_to_data = './data/Neuron_Dataset/'

# load labels and types as dict
map_to_types = {}
with open(path_to_data + 'types.csv', 'r') as f:
    for line in f:
        # replace ':' with '_'
        line = line.strip().replace(':', '_').split(', ')
        map_to_types[line[0]] = 0 if line[1] == 'pyramidal' else 1



print(map_to_types)
 
# %%
# Create a dataset
class NeuralDataset(Dataset):
    def __init__(self, path_to_data, map_to_type):
        self.path_to_data = path_to_data + 'data/'
        self.list_of_files = os.listdir(self.path_to_data)
        self.map_to_type = map_to_type
        
    def __len__(self):
        return len(os.listdir(self.path_to_data))
    
    def __getitem__(self, idx):
        # load diagram
        file_name = self.list_of_files[idx]
        data = np.load(self.path_to_data + file_name)
        
        # get label
        label = self.map_to_type[file_name.split('.')[0]]
        
        return data, label
    
ds = NeuralDataset(path_to_data, map_to_types)


def collate_fn(batch):
    data, label = zip(*batch)
    
    # pad data to max length with zeros in the zero-th dimension
    max_length = 902#max([len(d) for d in data])
    data = [np.pad(d, ((0, max_length - len(d)), (0, 0)), 'constant') for d in data]
    data = torch.tensor(data, dtype=torch.float32)
    label = np.array(label)
    label = torch.tensor(label, dtype=torch.long)
    return data, label

# Create a dataloader
dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

# %%
# Super simple model

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class Persformer(nn.Module):
    def __init__(
        self,
        dim_input=2,  # dimension of input data for each element in the set
        num_outputs=1,
        dim_output=2,  # number of classes
        num_inds=32,  # number of induced points, see  Set Transformer paper
        dim_hidden=128,
        num_heads=4,
        ln=False,  # use layer norm
        dropout=0.1,
        use_sab = False,
        num_encoder_layer=2,
        use_max_pool = False,
        use_attention_pool = True,
        use_sum_pool = False,
    ):
        super().__init__()
        
        num_pools = 0
        self.use_max_pool = use_max_pool
        self.use_attention_pool = use_attention_pool
        self.use_sum_pool = use_sum_pool
        if use_max_pool:
            num_pools += 1
        if use_attention_pool:
            num_pools += 1
        if use_sum_pool:
            num_pools += 1
        assert num_pools > 0, "At least one of the pooling methods must be used"
        
        if use_sab:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                *(SAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln) for _ in range(num_encoder_layer - 1))
            )
        else:
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                *(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln) for _ in range(num_encoder_layer - 1))
            )
        self.pool = nn.Sequential(
            nn.Dropout(dropout),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),   
        )
        self.dec = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(num_pools * dim_hidden, num_pools*dim_hidden),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(num_pools * dim_hidden, dim_hidden),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_output),
        )
        self.dim_hidden = dim_hidden

    def forward(self, input):
        out = self.enc(input)
        pool_list = []
        if self.use_max_pool:
            pool_list.append(out.max(1)[0])
        if self.use_attention_pool:
            pool_list.append(out.sum(dim=1))
        if self.use_sum_pool:
            pool_list.append(self.pool(out).reshape(-1, self.dim_hidden))
        return self.dec(
                torch.cat(
                        pool_list
                    , dim=1)
            ).squeeze()


model = Persformer(dim_input=2, dim_output=2, dropout=0.1, ln=True)
# %%
model(next(iter(dl))[0])
# %%

# initialise loss
loss_fn = nn.CrossEntropyLoss()

# initialise pipeline class
trainer = Trainer(model, [dl, None], loss_fn, writer)
# %%
trainer.train(Adam, n_epochs=100, cross_validation=False, optimizers_param={"lr": 0.0001})
# %%
sum = 0
for (data, label) in dl:
    sum += label.sum()
print('Dataset inbalance:', sum.item() / len(ds))
# %%
pipe = Trainer(
    model, [dl, None], loss_fn, writer,
)
# initialise gridsearch
search = HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)

# if you want to store pickle files of the models instead of the state_dicts
search.store_pickle = False

# dictionaries of hyperparameters
optimizers_params = {"lr": [0.001, 0.01]}
dataloaders_params = {"batch_size": [16, 32, 4],
                      collate_fn: [collate_fn]}
models_hyperparams = {
                      "num_inds": [32, 128 , 16],
                      "dim_hidden": [32, 128 , 16],
                      "num_heads": [4, 16, 4],
                      "ln": [False, True],
                      "dropout": [0.001, 0.01],
                      "use_sab":  [False, True],
                      "num_encoder_layer": [2, 4, 1],
                      "use_max_pool": [False, True],
                      "use_attention_pool":  [False, True],
                      "use_sum_pool":  [False, True],
                      }

# starting the HPO
search.start(
    Adam,
    3,
    False,
    optimizers_params,
    dataloaders_params,
    models_hyperparams,
)

# %%

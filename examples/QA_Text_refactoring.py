# %% [markdown]
# # Basic tutorial: Question answer
# #### Author: Matteo Caorsi
# 
# This short tutorial provides you with the basic functioning of *giotto-deep* API.
# 
# The example described in this tutorial is the one of question answer.
# 
# The main steps of the tutorial are the following:
#  1. creation of a dataset
#  2. creation of a model
#  3. define metrics and losses
#  4. train the model
#  5. extract some features of the network

# %%

import numpy as np

import torch
from torch import nn
from gdeep.data.transforming_dataset import TransformingDataset

from gdeep.models import FFNet

from gdeep.visualisation import  persistence_diagrams_of_activations

from torch.utils.tensorboard import SummaryWriter
from gdeep.data import TorchDataLoader
from gdeep.pipeline import Pipeline

from gtda.diagrams import BettiCurve

from gtda.plotting import plot_betti_surfaces

from gdeep.utility import autoreload_if_notebook

autoreload_if_notebook()

# %% [markdown]
# # Initialize the tensorboard writer
# 
# In order to analyse the reuslts of your models, you need to start tensorboard.
# On the terminal, move inside the `/example` folder. There run the following command:
# 
# ```
# tensorboard --logdir=runs
# ```
# 
# Then go [here](http://localhost:6006/) after the training to see all the visualisation results.

# %%
writer = SummaryWriter()

# %% [markdown]
# # Create your dataset

# %%
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.data import TorchDataLoader

# the only part of the training set we are interested in
train_indices = list(range(32*10))

dl = TorchDataLoader(name="SQuAD2", convert_to_map_dataset=True)
dl_tr_str, dl_ts_str = dl.build_dataloaders(sampler=SubsetRandomSampler(train_indices), batch_size=1)


# %% [markdown]
# The dataset contains a context and a question whose answer can be found within that context. The correct answer as well as the starting characters are also provided.

# %%
print("Before preprocessing: \n", dl_tr_str.dataset[0])


# %% [markdown]
# ## Required preprocessing
# 
# Neural networks cannot direcly deal with strings. We have first to preprocess the dataset in three main ways:
#  1. Tokenise the strings into its words
#  2. Build a vocabulary out of these words
#  3. Embed each word into a vector, so that each sentence becomes a list of vectors
# 
# The first two steps are performed by the `PreprocessTextQA`. The embedding will be added directly to the model.

# %%
from gdeep.data import PreprocessTextQA
from torch.utils.data import DataLoader
from gdeep.data.transforming_dataset import TransformingDataset

prec = PreprocessTextQA()
prec.fit_to_dataset(dl_tr_str.dataset)

textds = TransformingDataset(dl_tr_str.dataset, prec.transform)

textts = TransformingDataset(dl_ts_str.dataset, prec.transform)

# %%
print("After the preprocessing: \n", textds[0])

train_indices = list(range(64*2))
test_indices = list(range(64*1))

dl_tr = DataLoader(textds, batch_size=16, sampler=SubsetRandomSampler(train_indices))
dl_ts = DataLoader(textts, batch_size=16, sampler=SubsetRandomSampler(test_indices))

# %% [markdown]
# ## Define and train your model
# 
# The model for QA shall accept as input the context and the question and return the probabilities for the initial and final token of the answer in the input context. The output than, is a pair of logits.

# %%
from torch.nn import Transformer
from torch.optim import Adam, SparseAdam, SGD
import copy

# my simple transformer model
class QATransformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim):
        super(QATransformer, self).__init__()
        self.transformer = Transformer(d_model=embed_dim,
                                       nhead=2,
                                       num_encoder_layers=1,
                                       num_decoder_layers=1,
                                       dim_feedforward=512,
                                       dropout=0.1)
        self.embedding_src = nn.Embedding(src_vocab_size, embed_dim, sparse=True)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, embed_dim, sparse=True)
        self.generator = nn.Linear(embed_dim, 2)
        
    def forward(self, X):
        #print(X.shape)
        src = X[:,0,:]
        tgt = X[:,1,:]
        #print(src.shape, tgt.shape)
        src_emb = self.embedding_src(src)
        tgt_emb = self.embedding_tgt(tgt)
        #print(src_emb.shape, tgt_emb.shape)
        self.outs = self.transformer(src_emb, tgt_emb)
        #print(outs.shape)
        logits = self.generator(self.outs)
        #print(logits.shape)
        #out = torch.topk(logits, k=1, dim=2).indices.reshape(-1,44)
        #print(out, out.shape)
        return logits
    
    def __deepcopy__(self, memo):
        """this is needed to make sure that the 
        non-leaf nodes do not
        interfere with copy.deepcopy()
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def encode(self, src, src_mask):
        """this method is used only at the inference step"""
        return self.transformer.encoder(
                            self.embedding_src(src), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        """this method is used only at the inference step"""
        return self.transformer.decoder(
                          self.embedding_tgt(tgt), memory,
                          tgt_mask)

# %%
vocab_size = 1500000

src_vocab_size = vocab_size
tgt_vocab_size = vocab_size
emb_size = 64

model = QATransformer(src_vocab_size, tgt_vocab_size, emb_size)
print(model)

# %% [markdown]
# ## Define the loss function
# 
# This loss function is a adapted version of the Cross Entropy for the trnasformer architecture.

# %%

def loss_fn(output_of_network, label_of_dataloader):
    #print(output_of_network.shape, label_of_dataloader.shape)
    tgt_out = label_of_dataloader
    #print(tgt_out)
    logits = output_of_network
    cel = nn.CrossEntropyLoss()
    return cel(logits, tgt_out)


# %%
# prepare a pipeline class with the model, dataloaders loss_fn and tensorboard writer
pipe = Pipeline(model, (dl_tr, dl_ts), loss_fn, writer)

# train the model
pipe.train(SGD, 3, False, {"lr":0.01}, {"batch_size":16})

# %% [markdown]
# ## Answering questions!
# 
# Here we have a question and its associated context:

# %%
bb = next(iter(dl_ts_str))
bb[:2]

# %% [markdown]
# 
# Get the vocabulary and numericize the question and context to then input both to the model.

# %%
voc = prec.vocabulary
context = prec.tokenizer(bb[0][0])
question = prec.tokenizer(bb[1][0])

# get the indexes in the vocabulary of the tokens
context_idx = torch.tensor(list(map(voc.__getitem__,context)))
question_idx = torch.tensor(list(map(voc.__getitem__,question)))

# %%
aa = next(iter(dl_tr))
length_to_pad = aa[0].shape[-1]
pad_fn = lambda item : torch.cat([item, prec.pad_item * torch.ones(length_to_pad - item.shape[0])])

# these tansors are ready to be fitted into the model
context_ready_for_model = pad_fn(context_idx)
question_ready_for_model = pad_fn(question_idx)

# %% [markdown]
# Put the two tensors of context and question together and input them to the model

# %%
inp = torch.stack((context_ready_for_model, question_ready_for_model)).reshape(1,*aa[0].shape[1:]).long()
out = model(inp)

# %% [markdown]
# The output is the ligits for the start and end tokens of the answer. It is now time to extract them with `torch.argmax`

# %%
answer_idx = torch.argmax(out, dim=1)

try:
    if answer_idx[0][1] > answer_idx[0][0]:
        print("The model proposes: '", context[answer_idx[0][0]:answer_idx[0][1]],"...'")
    else:
        print("The model proposes: '", context[answer_idx[0][0]],"...'")
except IndexError:
    print("The model was not able to find the answer.")
print("The actual answer was: '" + bb[2][0][0]+"'")

# %% [markdown]
# # Extract inner data from your models

# %%
from gdeep.models import ModelExtractor

me = ModelExtractor(pipe.model, loss_fn)

lista = me.get_layers_param()

for k, item in lista.items():
    print(k,item.shape)


# %%
DEVICE = torch.device("cpu")
x = next(iter(dl_tr))[0]
pipe.model.eval()
pipe.model(x.to(DEVICE))

list_activations = me.get_activations(x)
len(list_activations)


# %%
x = next(iter(dl_tr))[0][0]
if x.dtype is not torch.int64:
    res = me.get_decision_boundary(x, n_epochs=1)
    res.shape

# %%
x, target = next(iter(dl_tr))
if x.dtype is torch.float:
    for gradient in me.get_gradients(x, target=target)[1]:
        print(gradient.shape)

# %% [markdown]
# # Visualise activations and other topological aspects of your model

# %%
from gdeep.visualisation import Visualiser

vs = Visualiser(pipe)

vs.plot_data_model()
#vs.plot_activations(x)
#vs.plot_persistence_diagrams(x)


# %%




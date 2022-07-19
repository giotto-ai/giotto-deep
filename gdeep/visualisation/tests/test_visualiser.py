import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from gdeep.visualisation import Visualiser
from gdeep.data.preprocessors import TokenizerTextClassification
from gdeep.data.datasets import DataLoaderBuilder, DatasetBuilder
from gdeep.trainer import Trainer
from gdeep.utility import DEVICE


bd = DatasetBuilder(name="AG_NEWS", convert_to_map_dataset=True)
ds_tr_str, ds_val_str, ds_ts_str = bd.build()


ptd = TokenizerTextClassification()

ptd.fit_to_dataset(ds_tr_str)
transformed_textds = ptd.attach_transform_to_dataset(ds_tr_str)  # type: ignore
transformed_textts = ptd.attach_transform_to_dataset(ds_val_str)  # type: ignore

# the only part of the training/test set we are interested in
train_indices = list(range(64*10))
test_indices = list(range(64*5))

dl_tr2, dl_ts2, _ = DataLoaderBuilder([transformed_textds,
                                      transformed_textts]).build([{"batch_size": 16,
                                                                   "sampler": SubsetRandomSampler(train_indices)},
                                                                  {"batch_size": 16,
                                                                   "sampler": SubsetRandomSampler(test_indices)}
                                                                  ])
writer = SummaryWriter()


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        mean = torch.mean(embedded, dim=1)
        return self.fc(mean)


def test_visualiser():
    vocab_size = len(ptd.vocabulary)
    emsize = 64
    loss_fn = nn.CrossEntropyLoss()
    model = TextClassificationModel(vocab_size, emsize, 4).to(DEVICE)
    pipe = Trainer(model, [dl_tr2, dl_ts2], loss_fn, writer)

    vs = Visualiser(pipe)

    x = next(iter(dl_tr2))
    vs.plot_interactive_model()
    vs.plot_3d_dataset()
    vs.plot_activations(x)
    vs.plot_persistence_diagrams(x)
    vs.plot_betti_surface_layers([0, 1], x)
    vs.plot_betti_curves_layers([0, 1], x)
    vs.plot_betti_numbers_layers(batch=x, filtration_value=0.3)

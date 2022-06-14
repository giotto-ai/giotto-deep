from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torch

from gdeep.data.datasets import DatasetBuilder
from gdeep.analysis.interpretability import Interpreter
from gdeep.visualisation import Visualiser
from gdeep.data import PreprocessingPipeline
from gdeep.data import TransformingDataset
from gdeep.data.preprocessors import Normalization, TokenizerTextClassification
from gdeep.data.datasets import DataLoaderBuilder
from gdeep.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# many time we get an IterableDataset which is good for memory consumption, but cannot be sampled!
# we can load the whole dataset in memory and sample it
bd = DatasetBuilder(name="AG_NEWS",
                    convert_to_map_dataset=True)
ds_tr_str, ds_val_str, ds_ts_str = bd.build()
ptd = TokenizerTextClassification()

ptd.fit_to_dataset(ds_tr_str)
transformed_textds = ptd.attach_transform_to_dataset(ds_tr_str)  # type: ignore
transformed_textts = ptd.attach_transform_to_dataset(ds_val_str)  # type: ignore

# the only part of the training/test set we are interested in
train_indices = list(range(64 * 10))
test_indices = list(range(64 * 5))

dl_tr2, dl_ts2, _ = DataLoaderBuilder((transformed_textds,  # type: ignore
                                       transformed_textts)).build(({"batch_size": 16,  # type: ignore
                                                                    "sampler": SubsetRandomSampler(train_indices)},
                                                                   {"batch_size": 16,
                                                                    "sampler": SubsetRandomSampler(test_indices)}
                                                                   ))


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


vocab_size: int = len(ptd.vocabulary)  # type: ignore
emsize: int = 64
# print(vocab_size, emsize)
model = TextClassificationModel(vocab_size, emsize, 4)
loss_fn = nn.CrossEntropyLoss()
pipe = Trainer(model, [dl_tr2, dl_ts2], loss_fn, writer)


def test_interpret_text() -> None:
    vs = Visualiser(pipe)
    inter = Interpreter(pipe.model,
                        method="LayerIntegratedGradients")

    inter.interpret_text("I am writing about money and business",
                         0,
                         ptd.vocabulary,  # type: ignore
                         ptd.tokenizer,  # type: ignore
                         layer=pipe.model.embedding,  # type: ignore
                         n_steps=500,
                         return_convergence_delta=True
                         )

    vs.plot_interpreter_text(inter)

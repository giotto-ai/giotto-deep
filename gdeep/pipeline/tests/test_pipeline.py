from torch import nn
from torch.optim import SGD
from sklearn.model_selection import StratifiedKFold
from gdeep.pipeline import Pipeline
from gdeep.models import FFNet
from gdeep.data import TorchDataLoader
from gdeep.search import GiottoSummaryWriter
import numpy as np
from gdeep.data import DataLoaderFromArray, TorchDataLoader
from gdeep.search import clean_up_files


@clean_up_files
def test_pipe_1():
    # model
    class model1(nn.Module):
        def __init__(self):
            super(model1, self).__init__()
            self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=[3, 5, 4]))
        def forward(self, x):
            return self.seqmodel(x).reshape(-1,2,2)

    model = model1()
    # dataloaders
    X = np.random.rand(100,3)
    y = np.random.randint(2,size=100*2).reshape(-1,2)
    dl = DataLoaderFromArray(X, y)
    dl_tr, dl_val, dl_ts = dl.build_dataloaders(batch_size=23)

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # tb writer
    writer = GiottoSummaryWriter()
    # pipeline
    pipe = Pipeline(model, [dl_tr, None],
                    loss_fn, writer)#,StratifiedKFold(5, shuffle=True))
    # then one needs to train the model using the pipeline!
    pipe.train(SGD, 2, True, {"lr": 0.001}, n_accumulated_grads=2)

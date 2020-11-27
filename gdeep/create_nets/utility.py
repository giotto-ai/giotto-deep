import torch
import numpy as np

from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *

import pandas as pd

def train_classification_nn(nn, X, y, lr=0.001, n_epochs=100, bs=32):
    """Train a neural network on a classifiction task

    Args:
    -----
     - nn (nn.Module): [description]
     - X 2darray: [description]
     - y 1darray: [description]
     - lr (float, optional): [description]. Defaults to 0.001.
     - epochs (int, optional): [description]. Defaults to 100.
     - bs (int): batch size
     
     Output:
     -------
      - fastai tabular learner
    """
    
    df=pd.DataFrame(X,columns=["x"+str(i) for i in range(len(X[0]))])
    df["label"]=[str(y1) for y1 in y]
    # creating databunch for training

    splits = RandomSplitter(valid_pct=0.2)(range_of(df))
    #valid_idx = range(len(df_fix)-200, len(df_fix)) # validation index
    y_names = 'label' # the dependent variable
    cat_names = [] #list of names of categorical variables
    cont_names = ["x"+str(i) for i in range(len(X[0]))] #continuous variables
    data = TabularDataLoaders.from_df(path='.',df=df,
                                      y_names=y_names,
                                      #valid_idx=valid_idx,
                                      bs=bs,
                                      lr=lr,
                                      shuffle_train=False,
                                      splits=splits,
                                      cont_names=cont_names)
    
    #creating learner
    learn = TabularLearner(data, model=nn, metrics=accuracy)#, loss_func=loss_fun)
    # training
    learn.fit_one_cycle(n_epochs)

    return learn


class SaveOutput:
    """[summary]
    """
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        pass

    def clear(self):
        self.outputs = []

    def get_outputs(self):
        return self.outputs

class SaveNodeOutput(SaveOutput):
    """[summary]

    Args:
        SaveOutput ([type]): [description]
    """
    def __init__(self, entry=0):
        super().__init__()
        self.entry = entry

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[:,self.entry].detach())

class SaveLayerOutput(SaveOutput):
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.detach())

class Layers_list:
    """ Layers_list
    """
    def __init__(self, layers):
        if isinstance(layers, str):
            if layers == "All":
                self.all = True
            else:
                raise ValueError
        elif isinstance(layers, list):
            self.layer_list = layers
            self.all = False
        else:
            raise ValueError

    def in_list(self, el):
        if self.all:
            return True
        else:
            return el in self.layer_list




def get_activations(model, X_tensor, layer_types=[torch.nn.Linear]):
    """Returns activation of layers

    Args:
        model ([type]): [description]
        X_tensor ([type]): [description]
        layer_types (list, optional): [description]. Defaults to [torch.nn.Linear].

    Returns:
        [type]: [description]
    """
    saved_output_layers = SaveLayerOutput()

    hook_handles = []

    for layer in model.modules():
        layer_in_layer_types = [isinstance(layer, layer_type)
                        for layer_type in layer_types]
        if any(layer_in_layer_types):
            handle = layer.register_forward_hook(saved_output_layers)
            hook_handles.append(handle)
    
    model.eval()

    model(None,X_tensor)

    for handle in hook_handles:
        handle.remove()

    return saved_output_layers

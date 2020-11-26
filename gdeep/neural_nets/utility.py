import torch
import numpy as np


def train_classification_nn(nn, X_tensor, y_tensor, lr=0.001, weight_decay=1e-5, n_epochs=1000, with_l2_reg=False):
    """Train a neural network on a classifiction task

    Args:
        nn (nn.Module): [description]
        X_tensor ([type]): [description]
        y_tensor ([type]): [description]
        lr (float, optional): [description]. Defaults to 0.001.
        weight_decay ([type], optional): [description]. Defaults to 1e-5.
        epochs (int, optional): [description]. Defaults to 1000.
        with_l2_reg (bool, optional): [description]. Defaults to False.
    """
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(nn.parameters(),
                    lr=lr, weight_decay=1e-5)


    for epoch in range(n_epochs):
        y_pred = nn(X_tensor)

        loss = loss_function(y_pred, y_tensor)

        # L2-regularization
        if with_l2_reg:
            lamb = 0.01
            l2_reg = torch.tensor(0.)
            for param in nn.parameters():
                l2_reg += torch.norm(param)
            loss += lamb*l2_reg

        if epoch%100 == 99:
            print(epoch, loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


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

    model(X_tensor)

    for handle in hook_handles:
        handle.remove()

    return saved_output_layers
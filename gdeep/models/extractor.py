from typing import List, Dict, Callable, Tuple, Union
from copy import copy

import torch

from ..analysis.decision_boundary import GradientFlowDecisionBoundaryCalculator
from ..analysis.decision_boundary import UniformlySampledPoint
from . import SaveLayerOutput
from gdeep.utility import DEVICE

from gdeep.utility.custom_types import Tensor


class ModelExtractor:
    """This class wraps nn.Modules to extract
    weights, activations, gradients and decision boundaries

    Args:
        model:
            standard torch module
        loss_fn :
            loss function

    Examples::

        from gdeep.analysis import Extractor
        # the candidate datum for a question-answering example
        x = next(iter(transformed_textts))[0].reshape(1, 2, -1).to(DEVICE)
        # the model extractor: you need a trainer and the loss function
        ex = ModelExtractor(trainer.model, loss_fn)
        # getting the names of the layers
        layer_names = ex.get_layers_param().keys()
        print("Let's extract the activations of the first attention layer: ", next(iter(layer_names)))
        self_attention = ex.get_activations(x)[:2]

    """

    def __init__(
        self, model: torch.nn.Module, loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> None:
        # self.model = model
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn

    def _send_to_device(
        self, x: Union[Tensor, List[Tensor]], y: Tensor
    ) -> Tuple[Tensor, Union[Tensor, List[Tensor]], Tensor]:
        """use this private method to send the
        ``x`` and ``y`` to the ``DEVICE``.

        Args:
            x:
                the input of the model, either a List[Tensor] or a Tensor
            y:
                the label

        Returns:
            (Tensor, Union[Tensor, List[Tensor]], Tensor)
                the prediction for x, x and the label

        """
        new_x: List[Tensor] = []
        if isinstance(x, tuple) or isinstance(x, list):
            for xi in x:
                try:
                    xi.requires_grad = True
                except RuntimeError:  # only float and complex tensors can have grads
                    pass
                new_x.append(xi.to(DEVICE))
            x = copy(new_x)

        else:
            try:
                x.requires_grad = True
            except RuntimeError:  # only float and complex tensors can have grads
                pass
            x = x.to(DEVICE)

        y = y.to(DEVICE)
        # Compute prediction and loss
        if isinstance(x, tuple) or isinstance(x, list):
            prediction = self.model(*x)
        else:
            prediction = self.model(x)
        return prediction, x, y

    def get_decision_boundary(
        self, input_example: Tensor, n_epochs: int = 100, precision: float = 0.1
    ) -> Tensor:
        """Compute the decision boundary for self.model
        with self.loss_fn

        Args:
            n_epochs:
                number of training cycles to find
                the decision boundary
            input_example:
                an example of a single input,
                to extract the dimensions of the feature space
            precision:
                parameter to filter the spurious data that are
                close to te decision boundary, but not close enough

        Returns:
            Tensor:
                the pointcloud defining the decision
                boundary with dimensions `(n_samples, n_features)`
        """
        input_dim = input_example.flatten().shape[0]
        bounding_box = [(-1, 1) for _ in range(input_dim)]
        us = UniformlySampledPoint(bounding_box, n_samples=3000)
        sample_points_tensor = torch.from_numpy(us()).to(torch.float)
        # reshape points as example
        sample_points_tensor = sample_points_tensor.reshape(
            -1, *input_example.shape
        ).to(DEVICE)

        sample_points_tensor = sample_points_tensor.to(DEVICE)
        # print(sample_points_tensor.shape)
        # Using new gradient flow implementation
        self.model.train()
        gf = GradientFlowDecisionBoundaryCalculator(
            model=self.model,
            initial_points=sample_points_tensor,
            optimizer=lambda params: torch.optim.Adam(params),
        )
        gf.step(number_of_steps=n_epochs)
        res = gf.get_filtered_decision_boundary(delta=precision).detach()
        return res

    def get_activations(self, x: Union[Tensor, List[Tensor]]) -> List[Tensor]:
        """Compute the activations of self.model with input
        `X`

        Args:
            x :
                an example of an input or
                an input batch of which to compute the activation(s)

        Returns:
            list:
                list of the activation Tensors
        """

        saved_output_layers = SaveLayerOutput()

        hook_handles = []
        for layer in self.model.modules():
            handle = layer.register_forward_hook(saved_output_layers)
            hook_handles.append(handle)
        self.model.eval()
        _, x, _ = self._send_to_device(
            x, torch.rand((1, 1))
        )  # here we also run the model(x)

        for handle in hook_handles:
            handle.remove()

        self.model.train()
        return saved_output_layers.get_outputs()

    def get_layers_param(self) -> Dict[str, Tensor]:
        """Returns parameters of layers

        Returns:
            dict:
                dict of tensors, corresponding
                to the layer parameters. The key of
                the dict is the name of the parameters
        """

        # layer_data = dict()
        # for name, layer_param in self.model.named_parameters():
        #    layer_data[name]=layer_param.data
        return self.model.state_dict()

    def get_layers_grads(self) -> List[Tensor]:
        """Returns the gradients of each layer

        Returns:
            list:
                list of tensors, corresponding
                to the layer gradients (weights and biases!).
        """

        # layer_data = dict()
        # for name, layer_param in self.model.named_parameters():
        #    layer_data[name]=layer_param.data
        sdict = self.model.parameters()
        output = []
        for v in sdict:
            output.append(v.grad)
        return output

    def get_gradients(
        self, batch: Tuple[Union[Tensor, List[Tensor]], Tensor], **kwargs
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Returns the **averaged gradient** of the self.loss_fn.
        To specify the target variable (e.g. the true class),
        use the keywords argument `target=`

        Args:
            batch:
                a data batch on which to compute the average
                gradients. If the batch has one single item, then
                it will output the gradients for that single datum

        Returns:
            list,  list:
                the gradients for the inputs; the list of tensors,
                corresponding to the gradient of the weights.
        """
        x, target = batch
        pred, x, target = self._send_to_device(x, target)
        loss = self.loss_fn(pred, target, **kwargs)

        for k, param in self.model.state_dict().items():
            if param.dtype is torch.float:
                param.requires_grad = True
                param.retain_grad()  # this hook allows to store grad
        loss.backward()
        output_grads = []
        for param in self.model.parameters():
            if param.dtype is torch.float:
                output_grads.append(param.grad)
        grads: List[Tensor] = []
        if isinstance(x, tuple) or isinstance(x, list):
            for xi in x:
                grads.append(xi.grad)  # type: ignore
        else:
            grads.append(x.grad)  # type: ignore
        return grads, output_grads

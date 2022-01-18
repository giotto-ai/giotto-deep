import torch

from ..analysis.decision_boundary import GradientFlowDecisionBoundaryCalculator
from ..analysis.decision_boundary import UniformlySampledPoint
from . import SaveLayerOutput

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class ModelExtractor:
    """This class wraps nn.Modules to extract
    weights, activations, gradients and decision boundaries

    Args:
        model (nn.Module):
            standard torch module
        loss_fn (Callable):
            loss function
    """

    def __init__(self, model, loss_fn):
        # self.model = model
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn

    def get_decision_boundary(self, input_example, n_epochs=100,
                              precision=0.1):
        """Compute the decision boundary for self.model
        with self.loss_fn
        
        Args:
            n_epochs (int):
                number of training cycles to find
                the decision boundary
            input_example (Tensor):
                an example of a single input,
                to extract the dimensions of the feature space

        Returns:
            Tensor:
                the pointcloud defining the decision
                boundary with dimensions `(n_samples, n_features)`
        """
        input_dim = input_example.flatten().shape[0]
        bounding_box = [(-1, 1) for _ in range(input_dim)]
        us = UniformlySampledPoint(bounding_box,
                                   n_samples=3000)
        sample_points_tensor = torch.from_numpy(us()).to(torch.float)
        # reshape points as example
        sample_points_tensor = \
            sample_points_tensor.reshape(-1, *input_example.shape).to(DEVICE)
        
        sample_points_tensor = sample_points_tensor.to(DEVICE)
        # print(sample_points_tensor.shape)
        # Using new gradient flow implementation
        self.model.train()
        gf = GradientFlowDecisionBoundaryCalculator(model=self.model,
                                                    initial_points=
                                                    sample_points_tensor,
                                                    optimizer=lambda params:
                                                    torch.optim.Adam(params))
        gf.step(number_of_steps=n_epochs)
        res = gf.get_filtered_decision_boundary(delta=
                                                precision).detach()
        return res

    def get_activations(self, X):
        """Compute the activations of self.model with input
        `X`

        Args:
            input_example (Tensor):
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
        self.model(X.to(DEVICE))


        for handle in hook_handles:
            handle.remove()

        self.model.train()
        return saved_output_layers.get_outputs()

    def get_layers_param(self) -> dict:
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
        
    def get_layers_grads(self) -> dict:
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

    def get_gradients(self, x, **kwargs) -> tuple:
        """Returns the **averaged gradient** of the self.loss_fn.
        To specify the target variable (e.g. the true class),
        use the keywords argument `target=`

        Args:
            x (Tensor):
                point at which to compute grad

        Returns:
            tensor:
                the gradients
            list:
                list of tensors, corresponding
                to the gradient of the weights.
        """
        
        x.requires_grad = True
        x = x.to(DEVICE)
        kwargs["target"] = kwargs["target"].to(DEVICE)
        loss = self.loss_fn(self.model(x),
                            **kwargs)

        for k, param in self.model.state_dict().items():
            if param.dtype is torch.float:
                param.requires_grad = True
                param.retain_grad()  # this hook allows to store grad
        loss.backward()
        output_grads = []
        for param in self.model.parameters():
            if param.dtype is torch.float:
                output_grads.append(param.grad)
        return x.grad, output_grads

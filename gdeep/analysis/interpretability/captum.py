from typing import Any, Tuple, Dict, Callable, List, Optional
import inspect

# from captum.attr import *
from captum.attr import TokenReferenceBase, visualization
import torch
from torch import nn
from .attribution_factory import get_attr

from gdeep.utility import DEVICE

Tensor = torch.Tensor


class Interpreter:
    """Class to visualise the activation maps,
    the attribution maps and saliency maps using
    different techniques.

    Args:
        model:
            the standard pytorch model
        method:
            the interpretability method. Find
            more info at https://captum.ai/tutorials/

    """
    x: Tensor
    attribution: Tensor
    sentence: str
    y: Tensor
    layer: Optional[torch.nn.Module]
    features_list: Optional[List]
    attribution_list: Optional[List]

    def __init__(self, model: nn.Module,
                 method: str = "IntegratedGradients"):
        # self.model = model
        self.model = model.to(DEVICE)
        self.method = method
        self.stored_visualisations: List = []

    def interpret(self,
                  x: Tensor,
                  y: Optional[Any] = None,
                  layer: Optional[torch.nn.Module] = None,
                  **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """This method creates a datum interpreter. This class
        is based on captum.

        Args:
            x:
                the tensor corresponding to the input datum.
                In case the datum is an image for example,
                it is expected to be of size ``(b, c, h, w)``
            y:
                the label we want to check the interpretability
                of.
            layer (nn.Module, optional):
                some methods will require to specify a layer
                of self.model

        Returns
            torch.Tensor, torch.Tensor:
                the input datum and the attribution
                image respectively.
        """
        self.x = x.to(DEVICE)
        self.layer = layer
        if self.layer:
            attr_class = get_attr(self.method, self.model, self.layer)
        else:
            attr_class = get_attr(self.method, self.model)
        self.model.eval()
        if y is not None:
            self.attribution = attr_class.attribute(self.x, target=y, **kwargs)
        else:
            self.attribution = attr_class.attribute(self.x, **kwargs)
        return self.x, self.attribution

    def feature_importance(self, x: Tensor, y: Tensor, **kwargs: Any) \
            -> Tuple[Tensor, List[Any]]:
        """This method creates a tabular data interpreter. This class
        is based on captum.

        Args:
            x:
                the datum
            y:
                the target label
            kwargs:
                kwargs for the attributions

        Returns:
            (Tensor, Tensor):
                returns x and its attribution

        """
        self.x = x.to(DEVICE)  # needed for attribution functions
        self.y = y.to(DEVICE)
        self.model.eval()
        attr_list = []
        attribute_dict = {"inputs": self.x, "target": self.y,
                          "sliding_window_shapes": (1, ),
                          "n_steps": 50, **kwargs
                          }
        self.features_list = ["IntegratedGradients", "IntegratedGradients",
                              "DeepLift", "FeatureAblation", "Occlusion"]
        for method in self.features_list:
            attribution_class = get_attr(method, self.model)
            args_names = inspect.signature(attribution_class.attribute).parameters.keys()
            kwargs_of_method = {k: v for k, v in attribute_dict.items() if k in args_names}
            attr_list.append(attribution_class.attribute(**kwargs_of_method))
        self.attribution_list = attr_list
        return self.x, self.attribution_list

    def interpret_text(self, sentence: str,
                       label: Any,
                       vocab: Dict[str, int],
                       tokenizer: Callable[[str], List[str]],
                       layer: Optional[torch.nn.Module] = None,
                       min_len: int = 7,
                       **kwargs) -> Tuple[str, Tensor]:
        """This method creates a text interpreter. This class
        is based on captum.

        Args:
            sentence :
                the input sentence
            label:
                the label we want to check the interpretability
                of.
            vocab :
                a ``gdeep.data.preprocessors`` vocabulary. Can
                be extracted via the ``vocabulary`` attribute.
            tokenizer :
                a ``gdeep.data.preprocessors`` tokenizer. Can
                be extracted via the ``tokenizer`` attribute.
            layer :
                torch module corresponding to the layer belonging to
                ``self.model``.
            min_len:
                minimum length of the text. Shorter texts are padded
            kwargs:
                additional arguments for the attribution

        """
        self.model.eval()
        self.sentence = sentence
        if layer:
            attr_class = get_attr(self.method, self.model, layer)
        else:
            attr_class = get_attr(self.method, self.model)
        text = tokenizer(sentence)
        if len(text) < min_len:
            text += [' '] * (min_len - len(text))
        indexed = [vocab[t] for t in text]

        self.model.zero_grad()

        input_indices = torch.tensor(indexed).to(DEVICE)
        input_indices = input_indices.unsqueeze(0)

        # input_indices dim: [sequence_length]
        seq_length = min_len

        # predict
        pred_temp = torch.softmax(self.model(input_indices), 1)
        pred = torch.max(pred_temp)
        pred_ind: float = torch.argmax(pred_temp).item()
        # generate reference indices for each sample
        pad_index = 0
        token_reference = TokenReferenceBase(reference_token_idx=pad_index)
        reference_indices = \
            token_reference.generate_reference(seq_length,
                                               DEVICE).unsqueeze(0)
        # compute attributions and approximation
        # delta using layer integrated gradients
        self.attribution, delta = attr_class.attribute(input_indices,
                                                       reference_indices,
                                                       target=label,
                                                       **kwargs)

        print('pred: ', pred_ind, '(', '%.2f' % pred, ')',
              ', delta: ', abs(delta.item()))

        self.add_attributions_to_visualizer(self.attribution, text, pred,
                                            pred_ind, label, delta.item(),
                                            self.stored_visualisations)
        return sentence, self.attribution

    @staticmethod
    def add_attributions_to_visualizer(attributions,
                                       text: List[str],
                                       pred: Tensor,
                                       pred_ind: float,
                                       label: Any,
                                       delta: Any,
                                       vis_data_records: List) -> None:
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        # storing couple samples in an array for visualization purposes
        vis_data_records.append(visualization.VisualizationDataRecord(
                                attributions,
                                pred,
                                str(pred_ind),
                                str(label),
                                "1",
                                attributions.sum(),
                                text,
                                delta))

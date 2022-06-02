from typing import Any, Tuple, Dict, Callable, List, Optional

from captum.attr import *
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

    def __init__(self, model: nn.Module,
                 method: str = "IntegratedGradients"):
        # self.model = model
        self.model = model.to(DEVICE)
        self.method = method
        self.stored_visualisations = []
        self.image = None
        self.x = None
        self.sentence = None
        self.attribution = None

    def interpret_image(self,
                        x: Tensor,
                        y: Any,
                        layer: Optional[torch.nn.Module] = None,
                        **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """This method creates an image interpreter. This class
        is based on captum.

        Args:
            x:
                the tensor corresponding to the input image.
                It is expected to be of size ``(b, c, h, w)``
            y:
                the label we want to check the interpretability
                of.
            layer (nn.Module, optional):
                some methods will require to specify a layer
                of self.model

        Returns
            torch.Tensor, torch.Tensor:
                the input image and the attribution
                image respectively.
        """
        self.x = x.to(DEVICE)
        attr_class = get_attr(self.method, self.model, layer)
        self.model.eval()
        self.attribution = attr_class.attribute(self.x, target=y, **kwargs)
        self.image = self.x
        return self.x, self.attribution

    def interpret_tabular(self, x: Tensor, **kwargs: Any) -> None:
        """This method creates a tabular interpreter. This class
        is based on captum.

        Args:
            x :
                the tensor corresponding to the input image.
                It is expected to be of size ``(b, c, h, w)``

        """
        self.x = x.to(DEVICE)  # needed for plotting functions
        self.model.eval()
        attr_class = get_attr(self.method, self.model)
        self.attribution = attr_class.attribute(self.x,
                                                **kwargs)
        # ig = IntegratedGradients(self.model)
        # ig_nt = NoiseTunnel(ig)
        # dl = DeepLift(self.model)
        # gs = GradientShap(self.model)
        # fa = FeatureAblation(self.model)
        # self.ig_attr_test = ig.attribute(self.x,
        #                                  n_steps=50,
        #                                  target=y,
        #                                  **kwargs)

    def interpret_text(self, sentence: str,
                       label: Any,
                       vocab: Dict[str, int],
                       tokenizer: Callable[[str], List[str]],
                       layer: torch.nn.Module,
                       min_len: int = 7) -> None:
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

        """
        self.model.eval()
        self.sentence = sentence
        attr_class = get_attr(self.method, self.model, layer)
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
        pred_ind = torch.argmax(pred_temp).item()
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
                                                       n_steps=500,
                                                       return_convergence_delta=True)

        print('pred: ', pred_ind, '(', '%.2f' % pred, ')',
              ', delta: ', abs(delta.item()))

        self.add_attributions_to_visualizer(self.attribution, text, pred,
                                            pred_ind, label, delta.item(),
                                            self.stored_visualisations)

    @staticmethod
    def add_attributions_to_visualizer(attributions,
                                       text: List[str],
                                       pred: Tensor,
                                       pred_ind: Tensor,
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

from captum.attr import * # TokenReferenceBase, \
    # visualization, FeatureAblation, \
    # DeepLift, NoiseTunnel, IntegratedGradients, \
    # LayerIntegratedGradients, Occlusion, \
    # GuidedGradCam, Saliency, InputXGradient
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class Interpreter:
    """Class to visualise the activation maps,
    the attribution maps and salicy maps using
    different techniques.

    Args:
        model (nn.Module):
            the standard pytorch model
        method (string):
            the interpretability method. Find
            more info at https://captum.ai/tutorials/

    """

    def __init__(self, model,
                 method="IntegratedGradients"):
        # self.model = model
        self.model = model.to(DEVICE)
        self.method = method
        self.stored_visualisations = []
        self.image = None
        self.X = None
        self.sentence = None
        self.attrib = None

    def interpret_image(self, X, y, layer = None, **kwargs):
        """This method creates an image interpreter. This class
        is based on captum.

        Args:
            X (tensor):
                the tensor corresponding to the input image.
                It is expected to be of size ``(b, c, h, w)``
            y (int or float or str)
                the label we want to check the interpretability
                of.
            layer (nn.Module, optional):
                some methods will require to specify a layer
                of self.model

        Returns
            (torch.Tensor, torch.Tensor):
                the input image and the attribution
                image respectively.
        """
        self.X = X.to(DEVICE)
        if self.method in ("GuidedGradCam",
                           "LayerConductance",
                           "LayerActivation",
                           "LayerGradCam",
                           "LayerDeepLift",
                           "LayerFeatureAblation",
                           "LayerIntegratedGradients",
                           "LayerGradientShap",
                           "LayerDeepLiftShap"):
            occlusion = eval(self.method+"(self.model, layer)")
        else:
            occlusion = eval(self.method+"(self.model)")
        self.model.eval()
        att = occlusion.attribute(self.X, target=y, **kwargs)
        self.image = self.X
        self.attrib = att
        return self.X, att

    def interpret_tabular(self, X_test, y, **kwargs):
        """This method creates an image interpreter. This class
        is based on captum.

        Args:
            X (tensor):
                the tensor corresponding to the input image.
                It is expected to be of size ``(b, c, h, w)``
            y (int or float or str)
                the label we want to check the interpretability
                of.

        Returns
            (torch.Tensor, torch.Tensor):
                the input image and the attribution
                image respectively.
        """
        self.X = X_test.to(DEVICE)  # needed for plotting functions
        y = y.to(DEVICE)
        self.model.eval()
        ig = IntegratedGradients(self.model)
        ig_nt = NoiseTunnel(ig)
        dl = DeepLift(self.model)
        # gs = GradientShap(self.model)
        fa = FeatureAblation(self.model)
        self.ig_attr_test = ig.attribute(self.X,
                                         n_steps=50,
                                         target=y,
                                         **kwargs)
        self.ig_nt_attr_test = ig_nt.attribute(self.X,
                                               target=y,
                                               **kwargs)
        self.dl_attr_test = dl.attribute(self.X,
                                         target=y,
                                         **kwargs)
        # self.gs_attr_test = gs.attribute(X_test, X_train, **kwargs)
        self.fa_attr_test = fa.attribute(self.X,
                                         **kwargs)

    def interpret_text(self, sentence, label, vocab,
                       tokenizer, layer,
                       min_len=7):
        """This method creates an image interpreter. This class
        is based on captum.

        Args:
            sentence (string):
                the input sentence
            label (int or float or str)
                the label we want to check the interpretability
                of.
            vocab (vocabulary):
                a ``gdeep.data.PreprocessText`` vocabulary. Can
                be extracted via the ``vocabulary``attribute.
            tokenizer (tokenizer):
                a ``gdeep.data.PreprocessText`` tokenizer. Can
                be extracted via the ``tokenizer``attribute.
            layer (nn.Module):
                torch module correspondign to the layer belonging to
                ``self.model``.

        Returns
            (torch.Tensor, torch.Tensor):
                the input image and the attribution
                image respectively.
        """
        self.model.eval()
        self.sentence = sentence
        lig = eval(self.method+"(" + ",".join(("self.model", #"self.model." +
                   "layer")) + ")")
        text = tokenizer(sentence)
        if len(text) < min_len:
            text += ['.'] * (min_len - len(text))
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
        PAD_IND = vocab['.']
        token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
        reference_indices = \
            token_reference.generate_reference(seq_length,
                                               DEVICE).unsqueeze(0)
        # compute attributions and approximation
        # delta using layer integrated gradients
        attributions_ig, delta = lig.attribute(input_indices,
                                               reference_indices,
                                               target=label,
                                               n_steps=500,
                                               return_convergence_delta=True)

        print('pred: ', pred_ind, '(', '%.2f' % pred, ')',
              ', delta: ', abs(delta.item()))

        self.add_attributions_to_visualizer(attributions_ig, text, pred,
                                            pred_ind, label, delta.item(),
                                            self.stored_visualisations,
                                            vocab)

    @staticmethod
    def add_attributions_to_visualizer(attributions, text, pred, pred_ind,
                                       label, delta, vis_data_records,
                                       vocab):
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

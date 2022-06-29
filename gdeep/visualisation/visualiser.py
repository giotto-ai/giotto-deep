from typing import List, Optional, Union, Tuple
import random
from copy import copy
import warnings

import torch
from torchvision.utils import make_grid
from sklearn.manifold import MDS
from gtda.diagrams import BettiCurve
from html2image import Html2Image
from captum.attr import visualization
import numpy as np
from captum.attr import LayerAttribution
from gtda.plotting import plot_diagram, plot_betti_surfaces, plot_betti_curves
import matplotlib.pyplot as plt

from gdeep.trainer import Trainer
from gdeep.analysis.interpretability import Interpreter
from gdeep.utility import DEVICE
from ..models import ModelExtractor
from . import (
    persistence_diagrams_of_activations,
    _simplified_persistence_of_activations,
    plotly2tensor,
    Compactification,
    png2tensor,
)

FONT_SIZE = 16
Tensor = torch.Tensor
Array = np.ndarray


class Visualiser:
    """This class is the bridge to send to the tensorboard
    the required plots. It converts plots and images of
    different kinds to tensors

    Args:
        pipe :
            the Trainer instance to get info from

    Examples::

        from gdeep.visualisation import Visualiser
        # you must have an initialised `trainer`
        vs = Visualiser(trainer)
        vs.plot_data_model()

    """
    persistence_diagrams: Optional[List[Array]] = None

    def __init__(self, pipe: Trainer) -> None:
        self.pipe = pipe

    def plot_interactive_model(self) -> None:
        """This function has no arguments: its purpose
        is to store the model to tensorboard for an
        interactive visualisation.
        """

        dataiter = iter(self.pipe.dataloaders[0])
        x, labels = next(dataiter)
        new_x: List[Tensor] = []
        if isinstance(x, tuple) or isinstance(x, list):
            for i, xi in enumerate(x):
                new_x.append(xi.to(DEVICE))
            x = copy(new_x)
        else:
            x = x.to(DEVICE)
        # add interactive model to tensorboard
        self.pipe.writer.add_graph(self.pipe.model, x)  # type: ignore
        self.pipe.writer.flush()  # type: ignore

    def plot_3d_dataset(self, n_pts: int = 100) -> None:
        """This function has no arguments: its purpose
        is to store data to the tensorboard for
        visualisation. Note that we are expecting the
        dataset item to be of type Tuple[Tensor, Tensor]

        Args:
            n_pts:
                number of points to display
        """
        dataiter = iter(self.pipe.dataloaders[0])
        features_list: List[Tensor] = []
        labels_list: List[str] = []

        for img, lab in dataiter:  # loop over batches
            if isinstance(img, tuple) or isinstance(img, list):
                features_list = features_list + [img[i][0] for i in range(len(img))]
            else:
                features_list = features_list + [img[i] for i in range(len(img))]
            try:
                labels_list = labels_list + [str(lab[i].item()) for i in range(len(lab))]
            except ValueError:
                labels_list = labels_list + ["Label not available" for _ in range(len(img))]
            if len(features_list) >= n_pts:
                break
        max_number = len(features_list)  # effective number of points

        features = torch.cat(features_list).to(DEVICE)

        if len(features.shape) >= 3:
            if len(features.shape) == 3:
                features = features.reshape(
                    max_number, -1, features.shape[1], features.shape[2]
                )
                grid = make_grid(features)
            else:
                grid = make_grid(features)

            self.pipe.writer.add_image("dataset", grid, 0)  # type: ignore

            self.pipe.writer.add_embedding(  # type: ignore
                features.view(max_number, -1),
                metadata=labels_list,
                label_img=features,
                tag="dataset",
                global_step=0,
            )
        else:
            self.pipe.writer.add_embedding(  # type: ignore
                features.view(max_number, -1),
                metadata=labels_list,
                tag="dataset",
                global_step=0,
            )

        self.pipe.writer.flush()  # type: ignore

    def plot_activations(self, batch: Optional[List[Union[List[Tensor], Tensor]]] = None) -> None:
        """Plot PCA of the activations of all layers of the
        `self.model`.

        Args:
            batch:
                this should be an input batch for the model
        """
        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        if batch is not None:
            inputs, labels = batch
        else:
            inputs, labels = next(iter(self.pipe.dataloaders[0]))

        acts = me.get_activations(inputs)
        print("Sending the plots to tensorboard: ")
        for i, act in enumerate(acts):
            print("Step " + str(i + 1) + "/" + str(len(acts)), end="\r")
            length = act.shape[0]
            self.pipe.writer.add_embedding(   # type: ignore
                act.view(length, -1),
                metadata=labels,
                tag="activations_" + str(i),
                global_step=0,
            )
            self.pipe.writer.flush()  # type: ignore

    def plot_persistence_diagrams(self, batch: Optional[List[Tensor]] = None,
                                  homology_dimensions: Optional[List[int]] = None,
                                  **kwargs) -> None:  # type: ignore
        """Plot a persistence diagrams of the activations

        Args:
            batch:
                a batch of data, in the shape of (datum, label)
            homology_dimensions:
                a list of the homology dimensions, like ``[0, 1]``
            kwargs:
                arguments for the ``persistence_diagrams_of_activations``
        """

        if homology_dimensions is None:
            homology_dimensions = [0, 1]
        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        if self.persistence_diagrams is None:
            if batch is not None:
                inputs, _ = batch
            else:
                inputs, _ = next(iter(self.pipe.dataloaders[0]))

            activ = me.get_activations(inputs)
            self.persistence_diagrams = persistence_diagrams_of_activations(activ,
                                                                            homology_dimensions=homology_dimensions,
                                                                            **kwargs)
        list_of_dgms = []
        for i, persistence_diagram in enumerate(self.persistence_diagrams):
            plot_persistence_diagram = plot_diagram(persistence_diagram)
            img_t = plotly2tensor(plot_persistence_diagram)
            list_of_dgms.append(img_t)
        features = torch.stack(list_of_dgms)
        # grid = make_grid(features)
        self.pipe.writer.add_images(  # type: ignore
            "Persistence diagrams of the activations", features, dataformats="NHWC"
        )
        self.pipe.writer.flush()  # type: ignore

    def plot_decision_boundary(self, compact: bool = False, n_epochs: int = 100, precision: float = 0.1):
        """Plot the decision boundary as the intrinsic
        hypersurface defined by ``self.loss_fn == 0.5``
        in the case of 2 classes. The method also works
        for an arbitrary number of classes. Note that
        we are expecting a model whose forward function
        has only one tensor as input (and not multiple
        arguments).

        Args:
            compact:
                if plotting the compactified
                version of the boundary
            n_epochs:
                number of training cycles to find
                the decision boundary
            precision:
                parameter to filter the spurious data that are
                close to te decision boundary, but not close enough

        """

        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        x = next(iter(self.pipe.dataloaders[0]))[0][0]

        if compact:
            # initlaisation of the compactification
            cc = Compactification(
                neural_net=self.pipe.model,
                precision=0.1,
                n_samples=500,
                epsilon=0.051,
                n_features=x.shape[0],  # if x is not List[Tensor]
                n_epochs=100,
            )

            d_final, label_final = cc.create_final_distance_matrix()
            embedding = MDS(n_components=3, dissimilarity="precomputed")
            db = embedding.fit_transform(d_final)
            self.pipe.writer.add_embedding(  # type: ignore
                db, tag="compactified_decision_boundary", global_step=0
            )
            self.pipe.writer.flush()  # type: ignore
            return db, d_final, label_final
        else:
            db = me.get_decision_boundary(x, n_epochs, precision)
            if len(db) > 0:
                self.pipe.writer.add_embedding(db, tag="decision_boundary", global_step=0)  # type: ignore
                self.pipe.writer.flush()  # type: ignore
            else:
                warnings.warn("No points sampled over the decision boundary")
            return db.cpu(), None, None

    def plot_betti_surface_layers(
            self, homology_dimensions: Optional[List[int]] = None,
            batch: Optional[Tensor] = None,
            **kwargs) -> None:  # type: ignore
        """
        Args:
            homology_dimensions :
                A list of
                homology dimensions, e.g. ``[0, 1, ...]``
            batch :
                the selected batch of data to
                compute the activations on. By
                defaults, this method takes the
                first batch of elements
            kwargs:
                optional arguments for the creation of
                persistence diagrams

        """
        if homology_dimensions is None:
            homology_dimensions = [0, 1]
        if self.persistence_diagrams is None:
            me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
            if batch is not None:
                inputs, _ = batch
            else:
                inputs = next(iter(self.pipe.dataloaders[0]))  # type: ignore
            self.persistence_diagrams = persistence_diagrams_of_activations(
                me.get_activations(inputs),
                homology_dimensions=homology_dimensions,
                **kwargs
            )
        bc = BettiCurve()
        dgms = bc.fit_transform(self.persistence_diagrams)

        plots = plot_betti_surfaces(
            dgms, samplings=bc.samplings_,
            homology_dimensions=homology_dimensions
        )

        for i in range(len(plots)):
            plots[i].show()

    def plot_betti_curves_layers(
            self, homology_dimensions: Optional[List[int]] = None,
            batch: Optional[Tensor] = None,
            **kwargs) -> None:  # type: ignore
        """
        Args:
            homology_dimensions :
                A list of
                homology dimensions, e.g. ``[0, 1, ...]``
            batch :
                the selected batch of data to
                compute the activations on. By
                defaults, this method takes the
                first batch of elements
            kwargs:
                optional arguments for the creation of
                persistence diagrams

        """
        if homology_dimensions is None:
            homology_dimensions = [0, 1]
        if self.persistence_diagrams is None:
            me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
            if batch is not None:
                inputs, _ = batch
            else:
                inputs = next(iter(self.pipe.dataloaders[0]))  # type: ignore
            self.persistence_diagrams = persistence_diagrams_of_activations(
                me.get_activations(inputs),
                homology_dimensions=homology_dimensions,
                **kwargs
            )
        bc = BettiCurve()
        bc_curves = bc.fit_transform(self.persistence_diagrams)
        list_of_plts = []
        for bc_curve in bc_curves:
            plots = plot_betti_curves(
                bc_curve, samplings=bc.samplings_,
                homology_dimensions=homology_dimensions
            )
            img_t = plotly2tensor(plots)
            list_of_plts.append(img_t)
        features = torch.stack(list_of_plts)
        self.pipe.writer.add_images(  # type: ignore
            "Betti curves for each layer", features, dataformats="NHWC"
        )
        self.pipe.writer.flush()  # type: ignore

    def plot_interpreter_text(self, interpreter: Interpreter):
        """This method allows to plot the results of an
        Interpreter for text data.

        Args:
            interpreter :
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
                
        Returns:
            matplotlib.figure
        """
        viz = interpreter.stored_visualisations
        fig = visualization.visualize_text(viz)
        name = "out.png"
        hti = Html2Image()
        hti.screenshot(html_str=fig.data, save_as=name)
        img_ten = png2tensor(name)
        self.pipe.writer.add_image(interpreter.method, img_ten, dataformats="HWC")  # type: ignore
        return fig

    def plot_interpreter_image(self, interpreter: Interpreter):
        """This method allows to plot the results of an
        Interpreter for image data.

        Args:
            interpreter:
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
                
        Returns:
            matplotlib.figure
        """

        try:
            attrib = (
                torch.permute(interpreter.attribution.squeeze().detach(), (1, 2, 0)).detach().cpu().numpy()
            )
        except ValueError:
            attrib = torch.permute(
                LayerAttribution.interpolate(
                    interpreter.attribution.detach().cpu(),
                    tuple(interpreter.x.squeeze().detach().cpu().shape[-2:]),
                ).squeeze(0),
                (1, 2, 0),
            )
            attrib = (
                torch.stack([attrib, attrib, attrib], dim=2).squeeze(-1).detach().cpu().numpy()
            )
        img = (
            torch.permute(interpreter.x.squeeze().detach(), (1, 2, 0)).detach().cpu().numpy()
        )

        fig, _ = visualization.visualize_image_attr_multiple(
            attrib,
            img,
            ["original_image", "heat_map", "blended_heat_map"],
            ["all", "all", "all"],
            show_colorbar=True,
        )
        self.pipe.writer.add_figure(interpreter.method, fig)  # type: ignore
        return fig

    def plot_feature_importance(self, interpreter: Interpreter):
        """This method allows to plot the results of an
        Interpreter for tabular data.

        Args:
            interpreter :
                this is a ``gdeep.analysis.interpretability``
                initialised ``Interpreter`` class
                
        Returns:
            matplotlib.figure
        """
        # prepare attributions for visualization
        x_test = interpreter.x
        x_axis_data = np.arange(x_test.shape[1])
        x_axis_data_labels = list(map(lambda idx: idx, x_axis_data))

        width = 0.14
        legends = interpreter.features_list

        fig = plt.figure(figsize=(20, 10))

        ax = plt.subplot()
        ax.set_title(
            "Comparing input feature importance across "
            + "multiple algorithms and learned weights"
        )
        ax.set_ylabel("Attributions")

        plt.rc("font", size=FONT_SIZE)  # fontsize of the text sizes
        plt.rc("axes", titlesize=FONT_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=FONT_SIZE)  # fontsize of the x and y labels
        plt.rc("legend", fontsize=FONT_SIZE - 4)  # fontsize of the legend
        for i in range(len(interpreter.attribution_list)):  # type: ignore
            attribution_sum = interpreter.attribution_list[i].detach().cpu().numpy().sum(0)  # type: ignore
            attribution_norm_sum = attribution_sum / np.linalg.norm(attribution_sum, ord=1)
            r = lambda: random.randint(0, 255)  # noqa
            color = ('#%02X%02X%02X' % (r(), r(), r()))
            ax.bar(
                x_axis_data,
                attribution_norm_sum,
                width,
                align="center",
                alpha=0.8,
                color=color,
            )

        ax.autoscale_view()
        plt.tight_layout()

        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels(x_axis_data_labels)

        plt.legend(legends, loc=3)
        self.pipe.writer.add_figure("Feature Importance", fig)  # type: ignore
        return plt

    def plot_attribution(self, interpreter: Interpreter, **kwargs) -> Tuple[plt.Figure, plt.Figure]:
        """this method generically plots the attribution of
        the interpreter

        Args:
            interpreter :
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
            kwargs:
                keywords arguments for the plot (visualize_image_attr of
                captum)

        Returns:
            matplotlib.figure, matplotlib.figure
                they correspond respectively to the datum and attribution"""
        t1 = Visualiser._adjust_tensors_to_plot(interpreter.x)
        t2 = Visualiser._adjust_tensors_to_plot(interpreter.attribution)
        fig1, _ = visualization.visualize_image_attr(t1, **kwargs)
        fig2, _ = visualization.visualize_image_attr(t2, **kwargs)
        self.pipe.writer.add_figure("Datum x", fig1)  # type: ignore
        self.pipe.writer.add_figure("Generic attribution of x", fig2)  # type: ignore
        return fig1, fig2

    def plot_self_attention(self, attention_tensor: List[Tensor], tokens_x: Optional[List[str]] = None,
                            tokens_y: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """This functions plots the self-attention layer of a transformer.

        Args:
            attention_tensor:
                list of the self-attetion tensors (i.e. the tensors
                corresponding to the activations, given an input
            tokens_x:
                The string tokens to be displayed along the
                x axis of the plot
            tokens_y:
                The string tokens to be displayed along the
                y axis of the plot

        Returns:
             Figure:
                 matplotlib pyplot
            """
        fig = plt.figure(**kwargs)

        for idx, scores in enumerate(attention_tensor):
            scores_np = Visualiser._adjust_tensors_to_plot(scores)
            ax = fig.add_subplot(4, 3, idx + 1)
            # append the attention weights
            im = ax.imshow(scores_np, cmap='viridis')

            fontdict = {'fontsize': 10}
            if tokens_x:
                ax.set_xticks(range(len(tokens_x)))
                ax.set_xticklabels(tokens_x, fontdict=fontdict, rotation=90)
            if tokens_y:
                ax.set_yticks(range(len(tokens_y)))
                ax.set_yticklabels(tokens_y, fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(idx + 1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()
        self.pipe.writer.add_figure("Self attention map", fig)  # type: ignore
        self.pipe.writer.flush()  # type: ignore
        return fig

    def plot_attributions_persistence_diagrams(self, interpreter: Interpreter, **kwargs) -> plt.Figure:
        """this method allows the plot, on top of the persistence
        diagram, of the attribution values. For example, this would
        be the method to call when you run saliency maps on the
        persistence diagrams. Note that all homology dimensions
        are plot together

        Args:
            interpreter :
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
            kwargs:
                keywords arguments for the plot (``matplotlib.pyplot``)

        Returns:
            matplotlib.figure.Figure
        """

        pd_x: Array = interpreter.x.detach().cpu().numpy()
        pd_attr: Array = interpreter.attribution.detach().cpu().numpy()

        yy: Array = (np.abs(pd_attr.T[0]) + np.abs(pd_attr.T[1])).flatten()
        yy = (yy - yy.min(axis=0)) / (yy.max(axis=0) - yy.min(axis=0))
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(x=pd_x.T[0], y=pd_x.T[1], c=yy, **kwargs)
        line = np.arange(0, max(pd_x.T[1]) * 1.1, 0.01)
        plt.plot(line, line, '--', color="black")
        plt.xlabel("Birth")
        plt.ylabel("Death")

        self.pipe.writer.add_figure("Persistence diagrams attributions", fig)  # type: ignore
        self.pipe.writer.flush()  # type: ignore

        return fig

    def plot_betti_numbers_layers(
            self, homology_dimensions: Optional[List[int]] = None,
            filtration_value: float = 1.,
            batch: Optional[Tensor] = None,
            **kwargs) -> plt.Figure:  # type: ignore
        """
        Args:
            homology_dimensions :
                A list of
                homology dimensions, e.g. ``[0, 1, ...]``
            filtration_value:
                the filtration value you want to threshold the
                filtration at
            batch :
                the selected batch of data to
                compute the activations on. By
                defaults, this method takes the
                first batch of elements
            kwargs:
                optional arguments for the creation of
                persistence diagrams

        """
        if homology_dimensions is None:
            homology_dimensions = [0, 1]

        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        if batch is not None:
            inputs, _ = batch
        else:
            inputs = next(iter(self.pipe.dataloaders[0]))  # type: ignore

        simplified_persistence_diagrams = _simplified_persistence_of_activations(me.get_activations(inputs),
                                                                                 homology_dimensions,
                                                                                 filtration_value, ** kwargs)

        betti_numbers = []

        for dgm in simplified_persistence_diagrams:
            xx = dgm[:, 2]
            betti_numbers.append(
                [len(xx[(dgm[:, 2] == dim) * (dgm[:, 1] == 1.)]) for dim in homology_dimensions]
            )
        betti_array = np.array(betti_numbers).T

        fig = plt.figure(figsize=(10, 6.8))
        for i, array in enumerate(betti_array):
            plt.plot(array, label=str(i) + "-th betti number")
        plt.xlabel("Layers")
        plt.ylabel("Betti numbers")
        plt.legend()

        self.pipe.writer.add_figure("Betti numbers over the layers", fig)  # type: ignore
        self.pipe.writer.flush()  # type: ignore

        return fig

    @staticmethod
    def _adjust_tensors_to_plot(tensor: Tensor) -> np.ndarray:
        """make sure that tensors that will be plotted as images have the
        correct dimensions"""
        if len(tensor.shape) == 4:
            tensor = tensor[0, :, :, :]
        elif len(tensor.shape) > 4:
            tensor = tensor[0, :, :, :, 0]
        list_of_permutation = torch.argsort(torch.tensor(tensor.shape)).tolist()
        list_of_permutation.reverse()
        tensor = tensor.permute(*list_of_permutation)

        if tensor.shape[-1] == 2:
            temporary = torch.zeros((tensor.shape[0], tensor.shape[1], 3))
            temporary[:, :, :2] = tensor
            tensor = temporary
        else:
            tensor = tensor[:, :, :4]
        return tensor.permute(1, 0, 2).cpu().detach().numpy()

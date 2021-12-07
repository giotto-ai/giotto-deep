import torch
from ..models import ModelExtractor
from torchvision.utils import make_grid
from gtda.plotting import plot_diagram, plot_betti_surfaces
from . import persistence_diagrams_of_activations, \
    plotly2tensor, Compactification, png2tensor
from sklearn.manifold import MDS
from gtda.diagrams import BettiCurve
from html2image import Html2Image
from captum.attr import visualization
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerAttribution

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class Visualiser:
    """This class is the bridge to send to the tensorboard
    the required plots. It converts plots and images of
    different kinds to tensors

    Args:
        pipe (Pipeline):
            the pipeline to get info from
        writer (tensorboard SummaryWriter):
            the tensorboard writer
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.persistence_diagrams = None

    def plot_data_model(self):
        """This function has no arguments: its purpose
        is to store data to the tensorboard for
        visualisation.
        """

        dataiter = iter(self.pipe.dataloaders[0])
        images, labels = dataiter.next()
        # print(str(labels.item()))
        self.pipe.writer.add_graph(self.pipe.model, images.to(DEVICE))
        features_list = []
        labels_list = []
        index = 0

        for img, lab in dataiter:
            index += 1
            features_list.append(img[0])
            try:
                labels_list.append(str(lab[0].item()))
            except ValueError:
                labels_list.append("Label not available")
            if index == 1000:
                break
        max_number = min(1000, len(labels_list))
        features = torch.cat(features_list).to(DEVICE)
        if len(features.shape) >= 3:
            if len(features.shape) == 3:
                features = features.reshape(max_number, -1,
                                            features.shape[1],
                                            features.shape[2])
                grid = make_grid(features)
            else:
                grid = make_grid(features)

            self.pipe.writer.add_image('dataset',
                                       grid,
                                       0)

            self.pipe.writer.add_embedding(features.view(max_number, -1),
                                           metadata=labels_list,
                                           label_img=features,
                                           tag='dataset',
                                           global_step=0)
        else:
            self.pipe.writer.add_embedding(features.view(max_number, -1),
                                           metadata=labels_list,
                                           tag='dataset',
                                           global_step=0)

        self.pipe.writer.flush()

    def plot_activations(self, example=None):
        """Plot PCA of the activations of all layers of the
        `self.model`
        """
        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        inputs = []
        labels = []
        for j, item in enumerate(self.pipe.dataloaders[0]):
            labels.append(item[1][0])
            inputs.append(item[0][0])
            if j == 100:
                break
        inputs = torch.stack(inputs)

        if example is not None:
            inputs = inputs.to(example.dtype)
        acts = me.get_activations(inputs)
        print("Sending the plots to tensorboard: ")
        for i, act in enumerate(acts):
            print("Step " + str(i+1) + "/" + str(len(acts)), end='\r')
            length = act.shape[0]
            self.pipe.writer.add_embedding(act.view(length, -1),
                                           metadata=labels,
                                           tag="activations_" + str(i),
                                           global_step=0)
            self.pipe.writer.flush()

    def plot_persistence_diagrams(self, example=None):
        """Plot a persistence diagrams of the activations
        """
        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        if self.persistence_diagrams is None:
            inputs = []
            for j, item in enumerate(self.pipe.dataloaders[0]):
                inputs.append(item[0][0])
                if j == 100:
                    break
            inputs = torch.stack(inputs)
            if example is not None:
                inputs = inputs.to(example.dtype)
            activ = me.get_activations(inputs)
            self.persistence_diagrams = \
                persistence_diagrams_of_activations(activ)
        list_of_dgms = []
        for i, persistence_diagram in enumerate(self.persistence_diagrams):
            plot_persistence_diagram = plot_diagram(persistence_diagram)
            imgT = plotly2tensor(plot_persistence_diagram)
            list_of_dgms.append(imgT)
        features = torch.stack(list_of_dgms)
        # grid = make_grid(features)
        self.pipe.writer.add_images("persistence_diagrams_of_activations",
                                    features, dataformats="NHWC")
        self.pipe.writer.flush()

    def plot_decision_boundary(self, compact=False):
        """Plot the decision boundary as the intrinsic
        hypersurface defined by self.loss_fn == 0.5

        Args:
            compact (bool):
                if plotting the compactified
                version of the boundary
        """

        me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
        X = next(iter(self.pipe.dataloaders[0]))[0][0]

        if compact:
            # initlaisation of the compactification
            cc = Compactification(precision=0.1,
                                  n_samples=500,
                                  epsilon=0.051,
                                  n_features=X.shape[0],
                                  n_epochs=100,
                                  neural_net=self.pipe.model)

            d_final, label_final = cc.create_final_distance_matrix()
            embedding = MDS(n_components=3,
                            dissimilarity="precomputed")
            db = embedding.fit_transform(d_final)
            self.pipe.writer.add_embedding(db,
                                           tag="compactified_decision_boundary",
                                           global_step=0)
            self.pipe.writer.flush()
            return db, d_final, label_final
        else:
            db = me.get_decision_boundary(X)
            self.pipe.writer.add_embedding(db,
                                           tag="decision_boundary",
                                           global_step=0)
            self.pipe.writer.flush()
            return db.cpu(), None, None

    def betti_plot_layers(self, homology_dimension=(0, 1), example=None):
        """
        Args:
            homology_dimension (int array, optional):
                An array of
                homology dimensions

        Returns:
           (None):
                a tuple of figures representing the Betti surfaces
                of the data accross layers of the NN, with one figure
                per dimension in homology_dimensions. Otherwise, a single
                figure representing the Betti curve of the single sample
                present.
        """

        if self.persistence_diagrams is None:
            me = ModelExtractor(self.pipe.model, self.pipe.loss_fn)
            inputs = self.pipe.dataloaders[0].dataset.data[:100]
            if example is not None:
                inputs = inputs.to(example.dtype)
            self.persistence_diagrams = \
                persistence_diagrams_of_activations(me.get_activations(inputs))
        BC = BettiCurve()
        dgms = BC.fit_transform(self.persistence_diagrams)

        plots = plot_betti_surfaces(dgms,
                                    samplings=BC.samplings_,
                                    homology_dimensions=homology_dimension)

        for i in range(len(plots)):
            plots[i].show()

    def plot_interpreter_text(self, interpreter):
        """This method allows to plot the results of an
        Interpreter for text data.

        Args:
            interpreter (Interpreter):
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
                
        Returns:
            matplotlib.figure
        """
        viz = interpreter.stored_visualisations
        fig = visualization.visualize_text(viz)
        name = "out.png"
        hti = Html2Image()
        hti.screenshot(
            html_str=fig.data,
            save_as=name
        )
        img_ten = png2tensor(name)
        self.pipe.writer.add_image(interpreter.method,
                                   img_ten,
                                   dataformats="HWC")
        return fig

    def plot_interpreter_image(self, interpreter):
        """This method allows to plot the results of an
        Interpreter for image data.

        Args:
            interpreter (Interpreter):
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
                
        Returns:
            matplotlib.figure
        """
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')],
                                                         N=256)
        try:
            attrib = np.transpose(interpreter.attrib.squeeze().detach().cpu().numpy(),
                                  (1, 2, 0))
        except ValueError:
            attrib = np.transpose(LayerAttribution.interpolate(interpreter.attrib.detach().cpu(),
                                                               tuple(interpreter.image.squeeze().detach().cpu().shape[-2:])).squeeze(0),
                                  (1, 2, 0))
            attrib = torch.stack((attrib, attrib, attrib),axis=2).squeeze(-1).detach().cpu().numpy()
        img = np.transpose(interpreter.image.squeeze().detach().cpu().numpy(),
                           (1, 2, 0))

        fig, _ = visualization.visualize_image_attr_multiple(
            attrib,
            img,
            ["original_image", "heat_map", "blended_heat_map"],
            ["all", "all", "all"],
            show_colorbar=True)
        self.pipe.writer.add_figure(interpreter.method,
                                    fig)
        return fig

    def plot_interpreter_tabular(self, interpreter):
        """This method allows to plot the results of an
        Interpreter for tabular data.

        Args:
            interpreter (Interpreter):
                this is a ``gdeep.analysis.interpretability``
                initilised ``Interpreter`` class
                
        Returns:
            matplotlib.figure
        """
        # prepare attributions for visualization
        X_test = interpreter.X
        x_axis_data = np.arange(X_test.shape[1])
        x_axis_data_labels = list(map(lambda idx: idx,
                                      x_axis_data))

        ig_attr_test_sum = interpreter.ig_attr_test.detach().cpu().numpy().sum(0)
        ig_attr_test_norm_sum = ig_attr_test_sum / \
            np.linalg.norm(ig_attr_test_sum, ord=1)

        ig_nt_attr_test_sum = \
            interpreter.ig_nt_attr_test.detach().cpu().numpy().sum(0)
        ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / \
            np.linalg.norm(ig_nt_attr_test_sum, ord=1)

        dl_attr_test_sum = \
            interpreter.dl_attr_test.detach().cpu().numpy().sum(0)
        dl_attr_test_norm_sum = dl_attr_test_sum / \
            np.linalg.norm(dl_attr_test_sum, ord=1)

        # gs_attr_test_sum = \
        #     interpreter.gs_attr_test.detach().cpu().numpy().sum(0)
        # gs_attr_test_norm_sum = gs_attr_test_sum / \
        #     np.linalg.norm(gs_attr_test_sum, ord=1)

        fa_attr_test_sum = interpreter.fa_attr_test.detach().cpu().numpy().sum(0)
        fa_attr_test_norm_sum = fa_attr_test_sum / \
            np.linalg.norm(fa_attr_test_sum, ord=1)

        # lin_weight = model.lin1.weight[0].detach().numpy()
        # y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

        width = 0.14
        legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'DeepLift',
                   'Feature Ablation', 'Weights']

        fig = plt.figure(figsize=(20, 10))

        ax = plt.subplot()
        ax.set_title('Comparing input feature importances across ' + \
            'multiple algorithms and learned weights')
        ax.set_ylabel('Attributions')

        FONT_SIZE = 16
        plt.rc('font', size=FONT_SIZE)  # fontsize of the text sizes
        plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

        ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center',
               alpha=0.8, color='#eb5e7c')
        ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width,
               align='center', alpha=0.7, color='#A90000')
        ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width,
               align='center', alpha=0.6, color='#34b8e0')
        # ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum,
        # width, align='center',  alpha=0.8, color='#4260f5')
        ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width,
               align='center', alpha=1.0, color='#49ba81')
        # ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width,
        # align='center', alpha=1.0, color='grey')
        ax.autoscale_view()
        plt.tight_layout()

        ax.set_xticks(x_axis_data + 0.5)
        ax.set_xticklabels(x_axis_data_labels)

        plt.legend(legends, loc=3)
        self.pipe.writer.add_figure("Feature Importance", fig)
        return plt

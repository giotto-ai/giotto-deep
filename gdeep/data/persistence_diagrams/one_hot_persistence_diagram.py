from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar

import numpy as np
import plotly.graph_objs as gobj
import torch
from gdeep.utility.utils import flatten_list_of_lists

T = TypeVar("T")
Tensor = torch.Tensor
Array = np.ndarray


class OneHotEncodedPersistenceDiagram:
    """This class represents a single one-hot encoded persistence diagram.
    
    Args:
        data:
            The data of the persistence diagram.
    """
    _data: Tensor
    _homology_dimension_names: List[str]

    def __init__(self, data: Tensor, homology_dimension_names: Optional[List[str]] = None):  # type: ignore
        _check_if_valid(data)
        self._data = data
        if homology_dimension_names is not None:
            self._homology_dimension_names = homology_dimension_names
        else:
            self._homology_dimension_names = ["H_" + str(i) for i in range(self.get_num_homology_dimensions())]
        assert len(self._homology_dimension_names) == self.get_num_homology_dimensions(), \
            "The number of names must be equal to the number of homology dimensions."

    def set_homology_dimension_names(self, homology_dimension_names: List[str]) -> None:
        """This method sets the homology dimension names.
        """
        assert len(homology_dimension_names) == self.get_num_homology_dimensions(), \
            "The number of names must be equal to the number of homology dimensions."
        self._homology_dimension_names = homology_dimension_names

    def get_num_homology_dimensions(self) -> int:
        """This method returns the number of homology dimensions.
        """
        return self._data.shape[-1] - 2

    def get_num_points(self) -> int:
        """This method returns the number of points.
        """
        return self._data.shape[0]

    def __repr__(self):
        return (f"OneHotEncodedPersistenceDiagram({self._data.shape})\n"
                f"{self.get_num_homology_dimensions()} homology dimensions\n"
                f"{self._data.shape[0]} points\ndata:\n {self._data.__repr__()}")

    def save(self, path: str) -> None:
        """This method saves the persistence diagram to a file.
        """
        torch.save(self._data, path)

    def get_points_in_homology_dimension(self, homology_dimension: int) -> \
            'OneHotEncodedPersistenceDiagram':
        """This method returns all points in a given homology dimension.
        """
        assert 0 <= homology_dimension < self.get_num_homology_dimensions(), \
            "The homology dimension must be smaller than the number of homology dimensions."
        return OneHotEncodedPersistenceDiagram(
            self._data[
                torch.where(
                    self._data[:, 2 + homology_dimension] == 1.0
                )
            ]
        )

    def get_raw_data(self) -> Tensor:
        """This method returns the raw data of the persistence diagram.
        This function should not be used to change the data.
        """
        return self._data

    @staticmethod
    def load(path: str) -> 'OneHotEncodedPersistenceDiagram':
        """This method loads a persistence diagram from a file.
        """
        return OneHotEncodedPersistenceDiagram(torch.load(path))

    def get_all_points_in_homology_dimension(self, homology_dimension: int) -> \
            'OneHotEncodedPersistenceDiagram':
        """This method returns all points in a given homology dimension.
        """
        assert homology_dimension < self.get_num_homology_dimensions(), \
            "The homology dimension must be smaller than the number of homology dimensions."
        return OneHotEncodedPersistenceDiagram(
            self._data[
                torch.where(
                    self._data[:, 2 + homology_dimension] == 1.0
                )
            ]
        )

    def plot(self, names: Optional[List[str]] = None) -> gobj.Figure:  # type: ignore
        """This method plots the persistence diagram.
        
        Args:
            names: 
                The names of the homology dimensions.
        
        Examples::

            pd =  torch.tensor\
                      ([[0.0928, 0.0995, 0.0000, 0.0000, 1.0000, 0.0000],
                        [0.0916, 0.1025, 1.0000, 0.0000, 0.0000, 0.0000],
                        [0.0978, 0.1147, 1.0000, 0.0000, 0.0000, 0.0000],
                        [0.0978, 0.1147, 0.0000, 0.0000, 1.0000, 0.0000],
                        [0.0916, 0.1162, 0.0000, 0.0000, 0.0000, 1.0000],
                        [0.0740, 0.0995, 1.0000, 0.0000, 0.0000, 0.0000],
                        [0.0728, 0.0995, 1.0000, 0.0000, 0.0000, 0.0000],
                        [0.0740, 0.1162, 0.0000, 0.0000, 0.0000, 1.0000],
                        [0.0728, 0.1162, 0.0000, 0.0000, 1.0000, 0.0000],
                        [0.0719, 0.1343, 0.0000, 0.0000, 0.0000, 1.0000],
                        [0.0830, 0.2194, 1.0000, 0.0000, 0.0000, 0.0000],
                        [0.0830, 0.2194, 1.0000, 0.0000, 0.0000, 0.0000],
                        [0.0719, 0.2194, 0.0000, 1.0000, 0.0000, 0.0000]])

            names = ["Ord0", "Ext0", "Rel1", "Ext1"]
            pd = OneHotEncodedPersistenceDiagram(pd, names)
            pd.plot()

        """
        names = self._homology_dimension_names if names is None else names
        assert len(names) == self.get_num_homology_dimensions(), \
            "The number of names must be equal to the number of homology dimensions."
        pd = self._data.detach().numpy()

        # convert one-hot encoding to categorical encoding
        pd_categorical = pd[:, 2:].argmax(axis=1)

        fig = _plot_diagram(np.concatenate([pd[:, :2], pd_categorical.reshape(-1, 1)], axis=1),
                            names=names)

        return fig

    def all_close(self, other: 'OneHotEncodedPersistenceDiagram',
                  atol: float = 1e-7) -> bool:
        """This method checks if the persistence diagrams are close.
        """
        for i in range(self.get_num_homology_dimensions()):
            if not torch.allclose(self.get_all_points_in_homology_dimension(i).get_raw_data(),
                                  other.get_all_points_in_homology_dimension(i).get_raw_data(),
                                  atol=atol):
                return False
        return True

    def get_lifetimes(self) -> Tensor:
        """This method returns the lifetimes of the points.
        """
        return self._data[:, 1] - self._data[:, 0]

    def filter_by_lifetime(self, min_lifetime: float, max_lifetime: float) -> \
            'OneHotEncodedPersistenceDiagram':
        """This method filters the persistence diagram by lifetime.
        
        Args:
            min_lifetime:
                The minimum lifetime of the remaining points.
            max_lifetime:
                The maximum lifetime of the remaining points.
        """
        lifetime: Tensor = self.get_lifetimes()
        return OneHotEncodedPersistenceDiagram(
            self._data[  # type: ignore
                torch.where(
                    (lifetime >= min_lifetime) & (lifetime <= max_lifetime)
                )
            ]
        )

    @staticmethod
    def from_numpy(data: Array) -> 'OneHotEncodedPersistenceDiagram':
        """This method creates a persistence diagram from a numpy array.
        """
        # assert data.dtype == np.float32, "The data must be of type np.float32. Otherwise,"\
        #     " the data will not be correctly converted."
        return OneHotEncodedPersistenceDiagram(torch.from_numpy(data.astype(np.float32)))


def collate_fn_persistence_diagrams(batch: List[Tuple[OneHotEncodedPersistenceDiagram, int]]) -> \
        Tuple[List[Tensor], Tensor]:
    """This function collates the data for the persistence diagram by padding the data, converting
    the data to tensors, converting the labels to tensors and generating masks for the valid
    entries.
    
    The input is a list of tuples of the form (persistence diagram, label).
    
    Args:
        batch:
            The list of tuples of the form (persistence diagram, label).
            
    Returns:
        The data, the labels and the masks.
    """
    max_num_points: int = max([len(x[0].get_raw_data()) for x in batch])
    num_homology_dimensions: int = batch[0][0].get_num_homology_dimensions()
    input_batch = torch.zeros(len(batch), max_num_points, num_homology_dimensions + 2)
    mask = torch.zeros(len(batch), max_num_points)
    for i, (pd, _) in enumerate(batch):
        input_batch[i, :len(pd.get_raw_data()), :] = pd.get_raw_data()
        mask[i, :len(pd.get_raw_data())] = 1.0

    label_batch = torch.tensor([x[1] for x in batch])

    return [input_batch, mask], label_batch


def _check_if_valid(data) -> None:
    """This method checks if the data is a valid persistence diagram.
    
    Args:
        data:
            The data to check.
            
    Raises:
        ValueError:
            If the data is not a valid persistence diagram.
    """

    if data.ndimension() != 2:
        raise ValueError("The input should be a 2-dimensional tensor."
                         f"The input has {data.ndimension()} dimensions.")
    assert data.shape[-1] > 2, \
        "The input should have at least one homology dimensions."
    assert torch.all(data[:, 2:] >= -1e-5) and \
           torch.allclose(data[:, 2:].sum(dim=1), torch.tensor(1.0)), \
           "The homology dimension should be one-hot encoded."


# def _sort_by_lifetime(data: Tensor) -> Tensor:
#     """This function sorts the points by their lifetime.
#     """
#     return data[(
#         data[:, 1] - data[:, 0]
#     ).argsort()]

def get_one_hot_encoded_persistence_diagram_from_gtda(persistence_diagram: Array) \
        -> OneHotEncodedPersistenceDiagram:
    """This function takes a single persistence diagram from giotto-tda and returns a one-hot encoded
    persistence diagram.
    
    Args:
        persistence_diagram:
            An array of shape (num_points, 3) where the first two columns
            represent the coordinates of the points and the third column
            represents the index of the homology dimension.
        
    Returns:
        OneHotEncodedPersistenceDiagram:
            A one-hot encoded persistence diagram. If the persistence diagram has only one homology
            dimension, the third column will be filled with ones.
    """
    assert persistence_diagram.ndim == 2 and persistence_diagram.shape[1] >= 2, \
        "The input should be a 2-dimensional array of shape (num_points, 3) or (num_points, 2)."

    if persistence_diagram.shape[1] == 2:
        return OneHotEncodedPersistenceDiagram(
            torch.stack((torch.tensor(persistence_diagram),
                         torch.ones(persistence_diagram.shape[0])), dim=1))
    else:
        homology_types: Set[int] = set([int(i) for i in persistence_diagram[:, 2]])
        type_to_one_hot_encoding: Dict[int, int] = {
            i: j for j, i in enumerate(homology_types)
        }
        one_hot_encoding: Tensor = torch.zeros(persistence_diagram.shape[0],
                                               len(homology_types))
        # TODO: Fill the one-hot encoding in a vectorized manner.
        for i, j in enumerate(persistence_diagram[:, 2]):
            one_hot_encoding[i, type_to_one_hot_encoding[int(j)]] = 1
        birth_death_diagram: Tensor = torch.tensor(persistence_diagram[:, :2])

        return OneHotEncodedPersistenceDiagram(
            torch.cat((birth_death_diagram,
                         one_hot_encoding), dim=1).float())


def get_one_hot_encoded_persistence_diagram_from_gudhi_extended(
        diagram: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) \
        -> OneHotEncodedPersistenceDiagram:
    """Convert an extended persistence diagram of a single graph to an
    array with one-hot encoded homology type.
    Args:
        diagram (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
            The diagram of an extended persistence of a single graph.
    
    Returns:
        np.ndarray:
            The diagram in one-hot encoded homology type of size
            (num_points, 6).
    """
    # Get the length of each array
    lengths = [len(array) for array in diagram]

    if lengths == [0, 0, 0, 0]:
        return OneHotEncodedPersistenceDiagram(torch.zeros((0, 6)))

    # One-hot encode the homology type
    homology_type: np.ndarray = np.array(
        flatten_list_of_lists(
            [[i] * lengths[i] for i in range(4)]
        )
    )
    homology_type_one_hot = np.zeros((sum(lengths), 4))
    homology_type_one_hot[np.arange(sum(lengths)), homology_type] = 1

    # Concatenate the arrays
    diagram_one_hot = np.concatenate([sub_diagram for sub_diagram in diagram],
                                     axis=0)
    diagram_one_hot = np.concatenate([diagram_one_hot, homology_type_one_hot],
                                     axis=1)

    return OneHotEncodedPersistenceDiagram.from_numpy(diagram_one_hot)


def _plot_diagram(diagram, names: List[str], homology_dimensions=None, plotly_params=None):
    """Plot a single persistence diagram.

    Args:
        diagram, ndarray of shape (n_points, 3):
            The persistence diagram to plot, where the third dimension along axis 1
            contains homology dimensions, and the first two contain (birth, death)
            pairs to be used as coordinates in the two-dimensional plot.
        homology_dimensions, list of int or None, optional, default: ``None``:
            Homology dimensions which will appear on the plot. If ``None``, all
            homology dimensions which appear in `diagram` will be plotted.
        plotly_params, dict or None, optional, default: ``None``:
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should be
            dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.
    Returns:
        fig :
            :class:`plotly.graph_objects.Figure` object;
            Figure representing the persistence diagram.
    """
    # TODO: increase the marker size
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    diagram = diagram[diagram[:, 0] != diagram[:, 1]]
    diagram_no_dims = diagram[:, :2]
    posinfinite_mask = np.isposinf(diagram_no_dims)
    neginfinite_mask = np.isneginf(diagram_no_dims)
    max_val = np.max(np.where(posinfinite_mask, -np.inf, diagram_no_dims))
    min_val = np.min(np.where(neginfinite_mask, np.inf, diagram_no_dims))
    parameter_range = max_val - min_val
    extra_space_factor = 0.02
    has_posinfinite_death = np.any(posinfinite_mask[:, 1])  # type: ignore
    if has_posinfinite_death:
        posinfinity_val = max_val + 0.1 * parameter_range
        extra_space_factor += 0.1
    extra_space = extra_space_factor * parameter_range
    min_val_display = min_val - extra_space
    max_val_display = max_val + extra_space

    fig = gobj.Figure()
    fig.add_trace(gobj.Scatter(
        x=[min_val_display, max_val_display],
        y=[min_val_display, max_val_display],
        mode="lines",
        line={"dash": "dash", "width": 1, "color": "black"},
        showlegend=False,
        hoverinfo="none"
    ))

    for dim in homology_dimensions:
        name = names[int(dim)]
        subdiagram = diagram[diagram[:, 2] == dim]
        unique, inverse, counts = np.unique(
            subdiagram, axis=0, return_inverse=True, return_counts=True
        )
        hovertext = [
            f"{tuple(unique[unique_row_index][:2])}" +
            (
                f", multiplicity: {counts[unique_row_index]}"
                if counts[unique_row_index] > 1 else ""
            )
            for unique_row_index in inverse
        ]
        y = subdiagram[:, 1]
        if has_posinfinite_death:
            y[np.isposinf(y)] = posinfinity_val  # type: ignore
        fig.add_trace(gobj.Scatter(
            x=subdiagram[:, 0], y=y, mode="markers",
            hoverinfo="text", hovertext=hovertext, name=name
        ))

    fig.update_layout(
        width=500,
        height=500,
        xaxis1={
            "title": "Birth",
            "side": "bottom",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
        },
        yaxis1={
            "title": "Death",
            "side": "left",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False, "scaleanchor": "x", "scaleratio": 1,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
        },
        plot_bgcolor="white"
    )

    # Add a horizontal dashed line for points with infinite death
    if has_posinfinite_death:
        fig.add_trace(gobj.Scatter(
            x=[min_val_display, max_val_display],
            y=[posinfinity_val, posinfinity_val],  # type: ignore
            mode="lines",
            line={"dash": "dash", "width": 0.5, "color": "black"},
            showlegend=True,
            name=u"\u221E",
            hoverinfo="none"
        ))

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig

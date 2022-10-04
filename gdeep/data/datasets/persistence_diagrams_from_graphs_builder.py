import os
from typing import List, Tuple
import ssl
import urllib
import zipfile
from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix

from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import (
    OneHotEncodedPersistenceDiagram,
)
from gdeep.utility.extended_persistence import graph_extended_persistence_hks
from gdeep.utility.constants import DEFAULT_GRAPH_DIR


PD = OneHotEncodedPersistenceDiagram


class PersistenceDiagramFromGraphBuilder:
    """
    This class is used to load the persistence diagrams of the graphs in a
    dataset. All graph datasets available at https://chrsmrrs.github.io/datasets/docs/datasets/
    are supported.
    """

    url: str = "https://www.chrsmrrs.com/graphkerneldatasets"

    def __init__(
        self,
        dataset_name: str,
        diffusion_parameter: float,
        root: str = DEFAULT_GRAPH_DIR,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_name:
                The name of the graph dataset to load, e.g. "MUTAG".
            diffusion_parameter:
                The diffusion parameter of the heat kernel
                signature. These are usually chosen to be as {0.1, 1.0, 10.0}.
            root:
                The directory where to store the dataset
        """
        self.dataset_name: str = dataset_name
        self.diffusion_parameter: float = diffusion_parameter
        self.num_homology_types: int = 4
        self.root: str = root
        self.output_dir: str = os.path.join(
            root,
            dataset_name + "_" + str(diffusion_parameter) + "_extended_persistence",
        )

    def create(self) -> None:
        # Check if the dataset exists in the specified directory
        if not os.path.exists(self.output_dir):
            print(f"Dataset {self.dataset_name} does not exist!")
            self._preprocess()
        else:
            print(
                f"Dataset {self.dataset_name} already exists!"
                " Skipping: dataset will not be created."
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the dataset.

        Returns:
            A string representation of the dataset.
        """
        return (
            f"{self.__class__.__name__}(dataset_name={self.dataset_name}, "
            f"diffusion_parameter={self.diffusion_parameter}, "
            f"root={self.root})"
        )

    def _preprocess(self) -> None:
        """
        Preprocess the dataset and save the persistence diagrams and the labels
        in the output directory.
        The persistence diagrams are computed using the heat kernel signature
        method and then each diagram is saved in a separate npy file in the
        diagrams subdirectory of the output directory.
        The labels are saved in a csv file in the output directory.

        """
        # Load the dataset
        self.graph_dataset = GraphDataset(
            root=DEFAULT_GRAPH_DIR, dataset_name=self.dataset_name, url=self.url
        )

        # Create the directory for PDs if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "diagrams"))
        else:
            # This should not be reached
            raise ValueError("Output directory already exists!")

        num_graphs = len(self.graph_dataset)

        labels: List[Tuple[int, int]] = []

        print("Computing the persistence diagrams...")
        for graph_idx, graph in tqdm(
            enumerate(self.graph_dataset), total=num_graphs  # type: ignore
        ):

            # Get the adjacency matrix
            adj_mat: np.ndarray = graph[0]

            # Compute the extended persistence
            persistence_diagram_one_hot = graph_extended_persistence_hks(
                adj_mat, diffusion_parameter=self.diffusion_parameter
            )
            # Sort the diagram by the persistence lifetime, i.e. the second
            # column minus the first column

            # Save the persistence diagram in a file
            persistence_diagram_one_hot.save(
                os.path.join(self.output_dir, "diagrams", f"{graph_idx}.npy")
            )

            # Save the label
            labels.append((graph_idx, graph[1]))

        # Save the labels in a csv file
        pd.DataFrame(labels, columns=["graph_idx", "label"]).to_csv(
            os.path.join(self.output_dir, "labels.csv"), index=False
        )


class GraphDataset(Dataset):
    """This class i a light weight built-i class to retrieve
    data from the TUDatasets https://chrsmrrs.github.io/datasets/docs/datasets/.

    Args:
        dataset_name:
            name of the dataset appearing in the TUDataset webpage
        root:
            the root folder where to store the .zip file
        url:
            the url at which to fetch the dataset, most likely of the form
            https://www.chrsmrrs.com/graphkerneldatasets/<datasetname.zip>
    """

    def __init__(self, dataset_name: str, root: str, url: str):
        self.dataset_name = dataset_name
        self.root = root
        self.url = url
        self.dataset: List[
            Tuple[np.ndarray, int]
        ] = []  # where to store the adj matrices in memory
        self._build_dataset()

    def _download_url(self, url: str, folder: str) -> str:
        """private method to download the .zip file of
        the dataset"""
        context = ssl._create_unverified_context()  # noqa
        data = urllib.request.urlopen(url, context=context)  # type: ignore # noqa
        if not os.path.exists(folder):
            os.makedirs(folder)
        path_to_zip = os.path.join(folder, self.dataset_name + ".zip")
        with open(path_to_zip, "wb") as f:
            # workaround for https://bugs.python.org/issue42853
            while True:
                chunk = data.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        return path_to_zip

    @staticmethod
    def _extract_zip(path_to_zip: str, folder: str) -> None:
        """Private method to extract the zip file of the dataset"""
        with zipfile.ZipFile(path_to_zip, "r") as f:
            f.extractall(folder)

    def _download(self) -> str:
        """private method to download and extract the dataset"""
        folder = os.path.join(self.root, self.dataset_name)
        path_to_zip = self._download_url(f"{self.url}/{self.dataset_name}.zip", folder)
        self._extract_zip(path_to_zip, folder)
        os.unlink(path_to_zip)
        return path_to_zip[:-4]
        # shutil.rmtree(self.raw_dir)
        # os.rename(os.path.join(folder, self.name), self.raw_dir)

    def _build_dataset(self) -> None:
        """private method to put the dataset in memory. The data is
        stored in ``self.dataset``"""
        path = self._download()  # location of the files

        idx_file = os.path.join(path, self.dataset_name + "_graph_indicator.txt")
        indices = np.loadtxt(idx_file, delimiter=",", dtype=np.int32).T - 1  # type: ignore

        graph_labels = os.path.join(path, self.dataset_name + "_graph_labels.txt")
        labels = (np.loadtxt(graph_labels, delimiter=",", dtype=np.int32).T + 1) // 2  # type: ignore

        adj_file = os.path.join(path, self.dataset_name + "_A.txt")
        sparse_array = np.loadtxt(adj_file, delimiter=",", dtype=np.int32).T - 1  # type: ignore

        rows: defaultdict = defaultdict(list)
        cols: defaultdict = defaultdict(list)

        for node_id in sparse_array[0]:
            rows[indices[node_id]] += [node_id]

        for node_id in sparse_array[1]:
            cols[indices[node_id]] += [node_id]

        for idx, _ in enumerate(rows):
            row = np.array(rows[idx])  # np.array([0, 3, 1, 0])
            col = np.array(cols[idx])  # np.array([0, 3, 1, 2])
            data = np.ones(len(row))  # np.array([4, 5, 7, 9])
            assert len(col) == len(row)
            shape_x = max(max(row) - min(row), max(col) - min(col)) + 1
            offset = min(min(row), min(col))
            # print(row, col, offset, shape_x)
            adj_mat = coo_matrix(
                (data, (row - offset, col - offset)), shape=(shape_x, shape_x)
            ).toarray()
            # print(adj_mat, idx)
            self.dataset.append((adj_mat, labels[idx]))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

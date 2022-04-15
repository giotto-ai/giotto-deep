
# %%
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.utils import to_dense_adj  # type: ignore

from gdeep.extended_persistence.gudhi_implementation import \
    graph_extended_persistence_gudhi
from gdeep.extended_persistence.heat_kernel_signature import \
    HeatKernelSignature
from gdeep.utility.constants import DEFAULT_GRAPH_DIR
# %%

dataset_name = "MUTAG"

# Load the dataset
dataset = TUDataset(root=DEFAULT_GRAPH_DIR,
                 name=dataset_name,
                 use_node_attr=False,
                 use_edge_attr=False)

# Get the first graph
graph = dataset[0]


# %%

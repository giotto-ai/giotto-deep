
# %%
from torch_geometric.datasets import TUDataset  # type: ignore

from gdeep.utility.constants import DEFAULT_GRAPH_DIR
# %%

dataset_name = "REDDIT-MULTI-12K"

# Download the dataset
data = TUDataset(root=DEFAULT_GRAPH_DIR,
                 name=dataset_name,
                 use_node_attr=False,
                 use_edge_attr=False)

# L
# %%

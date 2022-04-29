
# %%
# Autoreload modules
from gdeep.utility.utils import autoreload_if_notebook
autoreload_if_notebook()
# %%
# Import modules

from gdeep.data.graph_dataloaders import create_dataloaders
# %%

dataset_name: str = "PROTEINS"
diffusion_parameter: float = 10.0
batch_size: int = 32
test_size = 0.2

train_loader, test_loader = create_dataloaders(
    dataset_name=dataset_name,
    diffusion_parameter=diffusion_parameter,
    batch_size=batch_size,
    test_size=test_size,
)

batch = next(iter(train_loader))

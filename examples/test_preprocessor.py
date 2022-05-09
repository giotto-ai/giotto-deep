# %%
from gdeep.topology_layers import PersistenceDiagramFeatureExtractor
from gdeep.utility.constants import DEFAULT_DATA_DIR
import numpy as np

# Create persistence diagram with one homology dimension.
persistence_diagrams = np.random.rand(2, 10, 2)

mean = np.array([[0.5, 0.5]])
std = np.array([[0.1, 0.1]])

pd_extractor = PersistenceDiagramFeatureExtractor(
    mean=mean,
    std=std,
    number_of_homology_dimensions=1,
    number_of_most_persistent_features=3,
)


features = pd_extractor(persistence_diagrams)

# save the extractor to a file
pd_extractor.save_pretrained('.')

input_values = features['input_values']
attention_masks = features['attention_mask']
del pd_extractor
pd_extractor = PersistenceDiagramFeatureExtractor.from_pretrained('preprocessor_config.json')
# %%

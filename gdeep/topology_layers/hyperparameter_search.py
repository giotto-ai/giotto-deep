from typing import Union
from os.path import join
from json import load
from dotmap import DotMap  # type: ignore

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# Import the giotto-deep modules
from gdeep.topology_layers import Persformer
from gdeep.pipeline import Pipeline
from gdeep.search import Gridsearch, GiottoSummaryWriter
from gdeep.data import DlBuilderFromDataCloud

class HyperparameterSearch:
    """Hyperparametersearch for Persformer using cross-validation.

    Args:
        dataset_name (_type_): _description_
        hyperparameter_space (str): _description_
        path_cloud_dataset (str, optional): _description_. Defaults to None.
    """
    def __init__(self,
                 dataset_name,
                 writer,
                 hyperparameter_space: str,
                 path_cloud_dataset: Union[str, None]=None) -> None:
        if path_cloud_dataset is None:
            path_cloud_dataset = join("data", "DatasetCloud")
        
        self.writer = writer

        dl_cloud_builder = DlBuilderFromDataCloud(dataset_name,
                                        path_cloud_dataset)
        self.dl, _, _ = dl_cloud_builder.build_dataloaders()
        self.dataset_metadata = dl_cloud_builder.get_metadata()
        
        with open(hyperparameter_space) as f:
            hyperparameters_dicts = DotMap(load(f))
            self.dataloaders_params = hyperparameters_dicts.dataloaders_params
            self.models_hyperparams = hyperparameters_dicts.models_hyperparams
            self.optimizers_params = hyperparameters_dicts.optimizers_params
            schedulers_params = hyperparameters_dicts.schedulers_params
            
        model = Persformer.from_config(config_model, config_data)

    def get_dataset_metadata(self):
        return self.dataset_metadata
    
    def start(self):
        # initialize loss
        loss_fn = nn.CrossEntropyLoss()
        
        # starting the gridsearch
        search.start((eval(config_model.optimizer),),
                    n_epochs=schedulers_params.num_training_steps[0],
                    cross_validation=hyperparameters_dicts.cross_validation,
                    k_folds=hyperparameters_dicts.k_folds,
                    optimizers_params=optimizers_params,
                    dataloaders_params=dataloaders_params,
                    models_hyperparams=models_hyperparams, lr_scheduler=get_cosine_with_hard_restarts_schedule_with_warmup,
                    schedulers_params=schedulers_params)
from sympy import N
import torch
from torch.optim import AdamW, Adam, SGD
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup # type: ignore
from sklearn.model_selection import StratifiedKFold, KFold

from dotmap import DotMap  # type:ignore

from json import load

from optuna.pruners import NopPruner

from gdeep.topology_layers import Persformer
from gdeep.search import GiottoSummaryWriter, Gridsearch
from gdeep.data.datasets import DlBuilderFromDataCloud
from gdeep.pipeline import Pipeline

class PersformerHyperparameterSearch:
    """This class is used to perform hyperparameter search for Persfomer using 
    cross-validation. The search is performed using the Giotto-Deep GridSearch 
    class. The hyperparameter dictionaries are loaded from the metadata file.
    The search is performed on the dataset specified in the class constructor.
    The training data is downloaded from the DatasetCloud and the results are
    saved in the path provided in the constructor.
    
    Args:
        dataset_name (str):
            name of the dataset to be used for the 
            hyperparameter search. The dataset must be present in the
            DatasetCloud.
        download_directory (str):
            directory where the dataset is
            either downloaded or already present
        path_hpo_metadata (str):
            path to the metadata file containing the
            hyperparameter dictionaries specifying the search space and the
            search metric
        path_writer (str):
            path to the Tensorflow writer directory where the
            Tensorflow summaries are saved during the search process.
            
    Returns:
        None
    """ 
    def __init__(self,
                dataset_name,
                download_directory,
                path_hpo_metadata,
                path_writer) -> None:
        self.dataset_name = dataset_name
        self.download_directory = download_directory
        self.path_hpo_metadata = path_hpo_metadata
        self.path_writer = path_writer
        
    def _get_data_loader(self) -> torch.utils.data.DataLoader:
        """Returns the data loader for the dataset specified in the constructor.
        
        Returns:
            torch.utils.data.DataLoader:
            data loader for the dataset specified in the constructor
        """
        dl_cloud_builder = DlBuilderFromDataCloud(self.dataset_name,
                                   self.download_directory)

        # create the dataset from the downloaded dataset
        train_dataloader, _, _ = dl_cloud_builder\
            .build_dataloaders(batch_size=10)
        return train_dataloader
    
    
    def search(self) -> None:
        """Performs the hyperparameter search. The search is performed using
        the Giotto-Deep GridSearch class. The hyperparameter dictionaries
        are loaded from the metadata file. The search is performed on the
        dataset specified in the class constructor. The training data is
        downloaded from the DatasetCloud and the results are saved in the
        path provided in the constructor.
        
        Returns:
            None
        """
        
        model = Persformer()
        
        train_dataloader = self._get_data_loader()
        

        # initialize loss
        loss_fn = torch.nn.CrossEntropyLoss()

        # Initialize the Tensorflow writer
        writer = GiottoSummaryWriter(self.path_writer)
        
        # load hpo metadata
        with open(self.path_hpo_metadata) as config_data_file:
            hyperparameters_dicts = DotMap(load(config_data_file))
            dataloaders_params = hyperparameters_dicts.dataloaders_params
            models_hyperparams = hyperparameters_dicts.models_hyperparams
            optimizers_params = hyperparameters_dicts.optimizers_params
            schedulers_params = hyperparameters_dicts.schedulers_params

        # Initialize pipeline        
        pipe = Pipeline(model, [train_dataloader, None], loss_fn, writer,
                            eval(
                                hyperparameters_dicts.fold_mode + "(" +
                                str(hyperparameters_dicts.n_splits) + ", "+
                                "shuffle=" + str(hyperparameters_dicts.shuffle)
                                + ")"
                            )
                        )


        pruner = NopPruner()
        search = Gridsearch(pipe,
                            search_metric=hyperparameters_dicts.search_metric,
                            n_trials=hyperparameters_dicts.n_trials,
                            best_not_last=True,
                            pruner=pruner)

        # starting the hyperparameter search
        search.start([eval(opt) for opt in hyperparameters_dicts.optimizer],
                    n_epochs=schedulers_params.num_training_steps[0],
                    cross_validation=hyperparameters_dicts.cross_validation,
                    optimizers_params=optimizers_params,
                    dataloaders_params=dataloaders_params,
                    models_hyperparams=models_hyperparams,
                    lr_scheduler=eval(hyperparameters_dicts.scheduler[0]),
                    schedulers_params=schedulers_params)

        # Close the Tensorflow writer
        writer.close()
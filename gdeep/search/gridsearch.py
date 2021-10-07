import torch
import optuna
import pandas as pd
import numpy as np
from gdeep.utility import _are_compatible
from optuna.trial import TrialState
from gdeep.pipeline import Pipeline
from gdeep.search import Benchmark, _benchmarking_param
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.visualisation import plotly2tensor
from torch.optim import *
import plotly.express as px
from functools import partial

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class Gridsearch(Pipeline):
    """This is the generic class that allows
    the user to perform gridsearch over several
    parameters such as learning rate, optimizer.

    Args:
        obj (either Pipeline or Benchmark object):
            either a pipeline of a bechmark class
        search_metric (string):
            either 'loss' or 'accuracy'
        n_trials (int):
            number of total gridsearch trials

    """

    def __init__(self, obj, search_metric="loss", n_trials=10):
        self.is_pipe = None
        self.obj = obj
        self.bench = obj
        self.study = None
        self.metric = search_metric
        self.list_res = []
        self.df_res = None
        if (isinstance(obj, Pipeline)):
            super().__init__(self.obj.model,
                             self.obj.dataloaders,
                             self.obj.loss_fn,
                             self.obj.writer)
            # Pipeline.__init__(self, obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            self.is_pipe = True
        elif (isinstance(obj, Benchmark)):
            self.is_pipe = False

        self.search_metric = search_metric
        self.n_trials = n_trials
        self.val_epoch = 0
        self.train_epoch = 0


    def _objective(self, trial,
                   optimizers,
                   n_epochs,
                   cross_validation,
                   optimizers_params,
                   dataloaders_params,
                   models_hyperparams,
                   lr_scheduler,
                   scheduler_params,
                   writer_tag,
                   profiling,
                   k_folds):
        """default callback function for optuna's study
        
        Args:
            trial (optuna.trial):
                the independent variable
            optimizers (list of torch.optim):
                list of torch optimizers
            n_epochs (int):
                number of training epochs
            optimizers_param (dict):
                dictionary of the optimizers
                parameters, e.g. `{"lr": 0.001}`
            models_param (dict):
                dictionary of the model
                parameters
            lr_scheduler (torch.optim):
                a learning rate scheduler
            scheduler_params (dict):
                learning rate scheduler parameters
            writer_tag (string):
                tag to prepend to the ouput
                on tensorboard
            profiling (bool, default=False):
                whether or not you want to activate the
                profiler
            k_folds (int, default=5):
                number of folds in cross validation
        """

        # gegnerate optimizer
        optimizers_names = list(map(lambda x: x.__name__, optimizers))
        optimizer = eval(trial.suggest_categorical("optimizer", optimizers_names))
        
        # generate all the hyperparameters
        optimizers_param = self._suggest_params(trial, optimizers_params)
        dataloaders_param = self._suggest_params(trial, dataloaders_params)
        models_hyperparam = self._suggest_params(trial, models_hyperparams)
        # create a new model instance
        try:
            new_model = type(self.model)(**models_hyperparam)
        except TypeError:
            new_model = self.model
        new_pipe = Pipeline(new_model, self.dataloaders, self.loss_fn, self.writer)

        loss, accuracy = new_pipe.train(optimizer, n_epochs,
                                        cross_validation,
                                        optimizers_param,
                                        dataloaders_param,
                                        lr_scheduler,
                                        scheduler_params,
                                        (trial, self.search_metric),
                                        profiling,
                                        k_folds
                                        )
        self.writer.flush()
        # release resources
        del(new_pipe)
        del(new_model)
        if self.search_metric == "loss":
            return loss
        else:
            return accuracy

    def start(self,
              optimizers,
              n_epochs=1,
              cross_validation=False,
              optimizers_params=None,
              dataloaders_params=None,
              models_hyperparams=None,
              lr_scheduler=None,
              scheduler_params=None,
              writer_tag="model",
              profiling=False,
              k_folds=5):
        """method to be called when starting the gridsearch

        Args:
            optimizers (list of torch.optim):
                list of torch optimizers
            n_epochs (int):
                number of training epochs
            cross_validation (bool):
                whether or not to use cross-validation
            optimizers_params (dict):
                dictionary of optimizers params
            dataloaders_params (int):
                dictionary of dataloaders parameters
            models_hyperparams (dict):
                dictionary of model parameters
            lr_scheduler (torch.optim):
                torch learning rate schduler class
            scheduler_params (dict):
                learning rate scheduler parameters
            writer_tag (string):
                tag to prepend to the ouput
                on tensorboard
            profiling (bool, default=False):
                whether or not you want to activate the
                profiler
            k_folds (int, default=5):
                number of folds in cross validation
        """
        if self.search_metric == "loss":
            self.study = optuna.create_study(direction="minimize")
        else:
            self.study = optuna.create_study(direction="maximize")
        if self.is_pipe:
            # in the __init__, self.model and self.dataloaders are
            # already initialised. So they exist also in _objective()
            self._inner_optimisat_fun(self.model,self.dataloaders,
                                      optimizers,
                                      n_epochs,
                                      cross_validation,
                                      optimizers_params,
                                      dataloaders_params,
                                      models_hyperparams,
                                      lr_scheduler,
                                      scheduler_params,
                                      writer_tag,
                                      profiling,
                                      k_folds)

        else:
            self._inner_benchmark_fun(self.bench.models_dicts,
                                      self.bench.dataloaders_dicts,
                                      optimizers,
                                      n_epochs,
                                      cross_validation,
                                      optimizers_params,
                                      dataloaders_params,
                                      models_hyperparams,
                                      lr_scheduler,
                                      scheduler_params,
                                      writer_tag,
                                      profiling,
                                      k_folds)

        self._store_to_tensorboard()


    def _inner_optimisat_fun(self, model, dataloaders,
                             optimizers,
                             n_epochs,
                             cross_validation,
                             optimizers_params,
                             dataloaders_params,
                             models_hyperparams,
                             lr_scheduler,
                             scheduler_params,
                             writer_tag,
                             profiling,
                             k_folds):
        """private method to be decorated with the
        benchmark decorator to have benchmarking
        or simply used as is if no benchmarking is
        needed
        """
        
        writer_tag = "model"
        try:
            writer_tag = "Dataset: " + dataloaders["name"] + \
                " | Model: " + model["name"]
            super().__init__(model["model"],
                             dataloaders["dataloaders"],
                             self.bench.loss_fn,
                             self.bench.writer)
        except TypeError:
            pass

        self.study.optimize(lambda tr: self._objective(tr,
                                                       optimizers,
                                                       n_epochs,
                                                       cross_validation,
                                                       optimizers_params,
                                                       dataloaders_params,
                                                       models_hyperparams,
                                                       lr_scheduler,
                                                       scheduler_params,
                                                       writer_tag,
                                                       profiling,
                                                       k_folds),
                            n_trials=self.n_trials,
                            timeout=None)
        try:
            self._results(model_name = model["name"],
                     dataset_name = dataloaders["name"])
        except TypeError:
            self._results()


    def _inner_benchmark_fun(self, models_dicts,
                             dataloaders_dicts, *args, **kwargs):
        """Decorated function for benchmarking"""
        _benchmarking_param(self._inner_optimisat_fun,
                            [models_dicts,
                             dataloaders_dicts])(None, None,
                                                 *args, **kwargs)
        


    def _results(self, model_name = "model", dataset_name = "dataset"):
        """This class returns the dataframe with all the results of
        the gridsearch. It also saves the figures in the writer.

        Args:
            model_name (str):
                the model name for the
                tensorboard gridsearch table
            dataset_name (str)
                the dataset name for the
                tensorboard gridsearch table

        Returns:
            pd.DataFrame:
                the hyperparameter table
        """
        self.list_res = []
        trials = self.study.trials
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))
        try:
            print("Best trial:")
            trial_best = self.study.best_trial
            print("Metric Value for best trial: ", trial_best.value)
        except ValueError:
            pass
        
        for tria in trials:
            temp_list = []
            for val in tria.params.values():
                temp_list.append(val)
            self.list_res.append([model_name, dataset_name] + temp_list + [tria.value])

        self.df_res = pd.DataFrame(self.list_res, columns=["model", "dataset"] +
                              list(trial_best.params.keys())+[self.metric])

        # correlations of numercal coefficients
        list_of_arrays = []
        labels = []
        for col in self.df_res.columns:
            vals = self.df_res[col].values
            #print("type here", vals.dtype)
            if vals.dtype in [np.float16, np.float64, 
                              np.float32,
                              np.int32, np.int64,
                              np.int16]:
                list_of_arrays.append(vals)
                labels.append(col)
        #print(list_of_arrays)
        corr = np.corrcoef(np.array(list_of_arrays))
        
        try:
            fig2 = px.imshow(corr,
                    labels=dict(x="Parameters",
                                y="Parameters",
                                color="Correlation"),
                    x=labels,
                    y=labels
                   )
            fig2.update_xaxes(side="top")
            fig2.show()
            img2 = plotly2tensor(fig2)

            self.writer.add_images("Gridsearch correlation: " +
                                   model_name + " " + dataset_name,
                                   img2, dataformats="HWC")
            self.writer.flush()
        except ValueError:
            pass
        
        return self.df_res
        
    def _store_to_tensorboard(self):
        """Store the hyperparameters to tensorboard"""
        for i in range(len(self.df_res)):
            dictio = {k:(int(v) if isinstance(v, np.int64) else v) for k,v in dict(self.df_res.iloc[i][:-1]).items()}
            self.writer.add_hparams(dictio,
                                    {self.df_res.columns[-1]: self.df_res.iloc[i][-1]})
        
        self.writer.flush()
        
        return self.df_res

    @staticmethod
    def _suggest_params(trial, params):
        """Utility function to generate the parameters
        for the gridsearch. It is based on optuna `suggest_<type>`.

        Args:
            trial (optuna.trial):
                optuna trial variable
            params (dict):
                dictionary of parameters

        Returns:
            (dict):
                dictionary of selected parameters values
        """
        if params is None:
            return None
        param_temp = {k:trial.suggest_float(k,*v) for k,v in
                      params.items() if (type(v) is list or type(v) is tuple)
                      and (type(v[0]) is float or type(v[1]) is float)}
        param_temp2 = {k:trial.suggest_int(k,*v) for k,v in
                       params.items() if (type(v) is list or type(v) is tuple)
                       and (type(v[0]) is int or type(v[1]) is int)}
        param_temp.update(param_temp2)
        param = {k:trial.suggest_categorical(k, v) for k,v in
                 params.items() if (type(v) is list or type(v) is tuple)
                 and (type(v[0]) is str or type(v[1]) is str)}
        param.update(param_temp)
        
        return param


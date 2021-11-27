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
import warnings
from itertools import chain, combinations
from optuna.pruners import MedianPruner
from ..utility import save_model_and_optimizer

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
            either ``'loss'`` or ``'accuracy'``
        n_trials (int):
            number of total gridsearch trials
        best_not_last (bool):
            A flag to use the best validation accuracy over the
            epochs or the validation accuracy of the last epoch
        pruner (optuna.Pruners, default MedianPruner):
            Instance of an optuna pruner, can be user-defined

    """

    def __init__(self, obj, search_metric="loss", n_trials=10, best_not_last=False, pruner=None):
        self.best_not_last = best_not_last
        self.is_pipe = None
        self.study = None
        self.best_val_acc_gs = 0
        self.best_val_loss_gs = np.inf
        self.list_res = []
        self.df_res = None
        if (isinstance(obj, Pipeline)):
            self.pipe = obj
            super().__init__(self.pipe.model,
                             self.pipe.dataloaders,
                             self.pipe.loss_fn,
                             self.pipe.writer)
            # Pipeline.__init__(self, obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            self.is_pipe = True
        elif (isinstance(obj, Benchmark)):
            self.bench = obj
            self.is_pipe = False

        self.search_metric = (search_metric if search_metric in ("loss", "accuracy")
                              else None)
        if self.search_metric is None:
            raise ValueError("Wrong search_metric! "
                             "Either `loss` or `accuracy`")
        self.n_trials = n_trials
        self.val_epoch = 0
        self.train_epoch = 0
        if pruner is not None:
            self.pruner = pruner
        else:
            self.pruner = MedianPruner(n_startup_trials=5,
                                       n_warmup_steps=0,
                                       interval_steps=1,
                                       n_min_trials=1)

    def _initialise_new_model(self, models_hyperparam):
        """private method to find the maximal compatible set
        between models and hyperparameters
        
        Args:
            models_hyperparam (dict):
                model selected hyperparameters
        
        Returns:
            nn.Module
                torch nn.Module
        """
        
        list_of_params_keys = Gridsearch._powerset(list(models_hyperparam.keys()))
        list_of_params_keys.reverse()
        for params_keys in list_of_params_keys:
            sub_models_hyperparam = {k:models_hyperparam[k] for k in models_hyperparam.keys() if k in params_keys}
            try:
                #print(sub_models_hyperparam)
                new_model = type(self.model)(**sub_models_hyperparam)
                #print(new_model.state_dict())
                raise ValueError
            except TypeError:  # when the parameters do not match the model
                pass
            except ValueError:  # when the parameters match the model
                break

        try:
            new_model
        except NameError:
            warnings.warn("Model cannot be re-initialised. Using existing one.")
            new_model = self.model
        return new_model

    def _objective(self, trial,
                   optimizers,
                   n_epochs,
                   cross_validation,
                   optimizers_params,
                   dataloaders_params,
                   models_hyperparams,
                   lr_scheduler,
                   scheduler_params,
                   profiling,
                   k_folds,
                   parallel_tpu,
                   keep_training,
                   store_grad_layer_hist,
                   n_accumulated_grads,
                   writer_tag=""):
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
            profiling (bool):
                whether or not you want to activate the
                profiler
            k_folds (int, default=5):
                number of folds in cross validation
            parallel_tpu (bool):
                boolean value to run the computations
                on multiple TPUs
            keep_training (bool):
                This flag allows to restart a training from
                the existing optimizer as well as the
                existing model
            store_grad_layer_hist (bool):
                This flag allows to store the gradients
                and the layer values in tensorboard for
                each epoch
            n_accumulated_grads (int):
                this is the number of accumated grads. It
                is taken into account only for positive integers
            writer_tag (string):
                tag to prepend to the ouput
                on tensorboard
        """

        # generate optimizer
        optimizers_names = list(map(lambda x: x.__name__, optimizers))
        optimizer = eval(trial.suggest_categorical("optimizer", optimizers_names))

        # generate all the hyperparameters
        optimizers_param = Gridsearch._suggest_params(trial, optimizers_params)
        dataloaders_param = Gridsearch._suggest_params(trial, dataloaders_params)
        models_hyperparam = Gridsearch._suggest_params(trial, models_hyperparams)
        
        # tag for storing the results
        writer_tag += "/" + str(optimizers_param) + \
            str(dataloaders_param) + str(models_hyperparam)
        # create a new model instance
        self.model = self._initialise_new_model(models_hyperparam)
        self.pipe = Pipeline(self.model, self.dataloaders, self.loss_fn, self.writer)
        loss, accuracy = self.pipe.train(optimizer, n_epochs,
                                         cross_validation,
                                         optimizers_param,
                                         dataloaders_param,
                                         lr_scheduler,
                                         scheduler_params,
                                         (trial, self.search_metric),
                                         profiling,
                                         k_folds,
                                         parallel_tpu,
                                         keep_training,
                                         store_grad_layer_hist,
                                         n_accumulated_grads,
                                         writer_tag
                                         )
        best_loss = self.pipe.best_val_loss
        best_accuracy = self.pipe.best_val_acc
        self.writer.flush()
        # returns
        if self.search_metric == "loss":
            if self.best_not_last:
                self.best_val_acc_gs = max(self.best_val_acc_gs, best_accuracy)
                self.best_val_loss_gs = min(self.best_val_loss_gs, best_loss)
                return best_loss
            self.best_val_acc_gs = max(self.best_val_acc_gs, accuracy)
            self.best_val_loss_gs = min(self.best_val_loss_gs, loss)
            return loss
        else:
            if self.best_not_last:
                self.best_val_acc_gs = max(self.best_val_acc_gs, best_accuracy)
                self.best_val_loss_gs = min(self.best_val_loss_gs, best_loss)
                return best_accuracy
            self.best_val_acc_gs = max(self.best_val_acc_gs, accuracy)
            self.best_val_loss_gs = min(self.best_val_loss_gs, loss)
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
              profiling=False,
              k_folds=5,
              parallel_tpu=False,
              keep_training=False,
              store_grad_layer_hist=False,
              n_accumulated_grads:int=0,
              writer_tag=""):
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
            profiling (bool, default=False):
                whether or not you want to activate the
                profiler
            k_folds (int, default=5):
                number of folds in cross validation
            parallel_tpu (bool):
                boolean value to run the computations
                on multiple TPUs
            n_accumulated_grads (int, default=0):
                number of accumulated gradients. It is
                considered only if a positive integer
            writer_tag (str):
                tag to prepend to the ouput
                on tensorboard
        """
        if self.search_metric == "loss":
            self.study = optuna.create_study(direction="minimize",
                                             pruner=self.pruner)
        else:
            self.study = optuna.create_study(direction="maximize",
                                             pruner=self.pruner)
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
                                      profiling,
                                      k_folds,
                                      parallel_tpu,
                                      keep_training,
                                      store_grad_layer_hist,
                                      n_accumulated_grads,
                                      writer_tag)

        else:
            _benchmarking_param(self._inner_optimisat_fun,
                                [self.bench.models_dicts,
                                 self.bench.dataloaders_dicts],
                                optimizers,
                                n_epochs,
                                cross_validation,
                                optimizers_params,
                                dataloaders_params,
                                models_hyperparams,
                                lr_scheduler,
                                scheduler_params,
                                profiling,
                                k_folds,
                                parallel_tpu,
                                keep_training,
                                store_grad_layer_hist,
                                n_accumulated_grads,
                                writer_tag="")

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
                             profiling,
                             k_folds,
                             parallel_tpu,
                             keep_training,
                             store_grad_layer_hist,
                             n_accumulated_grads,
                             writer_tag=""):
        """private method to be decorated with the
        benchmark decorator to have benchmarking
        or simply used as is if no benchmarking is
        needed
        """
        
        
        try:
            writer_tag = "Dataset:" + dataloaders["name"] + \
                "|Model:" + model["name"]
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
                                                       profiling,
                                                       k_folds,
                                                       parallel_tpu,
                                                       keep_training,
                                                       store_grad_layer_hist,
                                                       n_accumulated_grads,
                                                       writer_tag),
                            n_trials=self.n_trials,
                            timeout=None)
        try:
            self._results(model_name = model["name"],
                     dataset_name = dataloaders["name"])
            save_model_and_optimizer(self.pipe.model,
                                     model["name"],
                                     self.pipe.optimizer)
        except TypeError:
            try: 
                self._results(model_name = self.pipe.model.__class__.__name__,
                     dataset_name = self.pipe.dataloaders[0].dataset.__class__.__name__)
                save_model_and_optimizer(self.pipe.model,
                                         optimizer=self.pipe.optimizer)
            except AttributeError:
                self._results()


    def _results(self, model_name="model", dataset_name="dataset"):
        """This method returns the dataframe with all the results of
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
            if self.search_metric == "loss":
                self.list_res.append([model_name, dataset_name] + temp_list + [tria.value, -1])
            else:
                self.list_res.append([model_name, dataset_name] + temp_list + [np.inf, tria.value])

        self.df_res = pd.DataFrame(self.list_res, columns=["model", "dataset"] +
                              list(trial_best.params.keys())+["loss", "accuracy"])
        # compute hyperparams correlaton
        corr, labels = self._correlation_of_hyperparams()
        
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
            warnings.warn("Cannot send the picture of the correlation" +
                          " matrix to tensorboard")
        
        return self.df_res

    def _correlation_of_hyperparams(self):
        """Correlations of numerical hyperparameters"""
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
        return corr, labels

    def _store_to_tensorboard(self):
        """Store the hyperparameters to tensorboard"""
        for i in range(len(self.df_res)):
            dictio = {k:(int(v) if isinstance(v, np.int64) else v) for k,v in dict(self.df_res.iloc[i][:-2]).items()}
            self.writer.add_hparams(dictio,
                                    {self.df_res.columns[-2]: self.df_res.iloc[i][-2],
                                     self.df_res.columns[-1]: self.df_res.iloc[i][-1]})
        
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
        param_temp = {}
        param_temp = {k:Gridsearch._new_suggest_float(trial, k,*v) for k,v in
                      params.items() if (isinstance(v, list) or isinstance(v, tuple))
                      and (isinstance(v[0], float) or isinstance(v[1], float))}
        param_temp2 = {k:trial.suggest_int(k,*v) for k,v in
                       params.items() if (isinstance(v, list) or isinstance(v, tuple))
                       and (isinstance(v[0], int) or isinstance(v[1], int))}
        param_temp.update(param_temp2)
        param = {k:trial.suggest_categorical(k, v) for k,v in
                 params.items() if (isinstance(v, list) or isinstance(v, tuple))
                 and (isinstance(v[0], str) or isinstance(v[1], str))}
        param.update(param_temp)
        #print(param)
        return param

    @staticmethod
    def _new_suggest_float(trial, name, low, high, step=None, log=False):
        """A modification of the Optuna function, in order to remove the `*`"""
        return trial.suggest_float(name, low, high, log=log, step=step)

    @staticmethod
    def _powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


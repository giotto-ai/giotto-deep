import torch
import optuna
import os
import pandas as pd
import numpy as np
import time
from gdeep.utility import _are_compatible, _inner_refactor_scalars
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import random
import string


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class GiottoSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict,
                    hparam_domain_discrete=None, run_name=None,
                    scalars_lists=None, best_not_last=False):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict):
                Each key-value pair in the dictionary is the
                name of the hyper parameter and it's corresponding value.
                The type of the value can be one of `bool`, `string`, `float`,
                `int`, or `None`.
            metric_dict (dict):
                Each key-value pair in the dictionary is the
                name of the metric and it's corresponding value. Note that the key used
                here should be unique in the tensorboard record. Otherwise the value
                you added by ``add_scalar`` will be displayed in hparam plugin. In most
                cases, this is unwanted.
            hparam_domain_discrete:
                (Optional[Dict[str, List[Any]]]) A dictionary that
                contains names of the hyperparameters and all discrete values they can hold
            run_name (str):
                Name of the run, to be included as part of the logdir.
                If unspecified, will use current timestamp.
            scalars_lists (list):
                The lists for the loss and accuracy plots.
                This is a list with two lists
                (one for accuracy and one for the loss).
                Each one of the inner lists contain the
                pairs (metric_value, epoch).

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with GiottoSummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if (not isinstance(hparam_dict, dict)) or (not isinstance(metric_dict, dict)):
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        if not run_name:
            run_name = str(time.time()).replace(":","-")
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            if isinstance(scalars_lists, list) or isinstance(scalars_lists, tuple):
                scalars_list_loss = scalars_lists[0]
                if best_not_last:
                    scalars_list_loss = [x for x in scalars_list_loss[:np.argmin(np.array(scalars_list_loss)[:,0])+1]]
                for v, t in scalars_list_loss:
                    w_hp.add_scalar("loss", v, t)
                scalars_list_acc = scalars_lists[1]
                if best_not_last:
                    scalars_list_acc = [x for x in scalars_list_acc[:np.argmax(np.array(scalars_list_acc)[:,0])+1]]
                for v, t in scalars_list_acc:
                    w_hp.add_scalar("accuracy", v, t)


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
        best_not_last (bool, default False):
            boolean flag that is ``True`` would use
            the best metric over epochs averaged over the folds in CV
            rather than the last value of the metrics
            over the epochs averaged over the folds
        pruner (optuna.Pruners, default MedianPruner):
            Instance of an optuna pruner, can be user-defined
        sampler (optuna.Samplers, default TPESampler):
            If left unspecified, ``TPESample`` is used during single-objective 
            optimization and ``NSGAIISampler`` during multi-objective optimization
        db_url (str):
            name of the database to connect to. For example
            ``mysql+mysqldb://usr:psw@host:port/db_name``
        study_name (str):
            name of the optuna study

    """

    def __init__(self, obj, search_metric="loss", n_trials=10,
                 best_not_last=False,
                 pruner=None, sampler=None,
                 db_url=None, study_name=None):
        self.best_not_last_gs = best_not_last
        self.is_pipe = None
        self.study = None
        self.best_val_acc_gs = 0
        self.best_val_loss_gs = np.inf
        self.list_res = []
        self.df_res = None
        self.db_url = db_url
        self.study_name = study_name
        if (isinstance(obj, Pipeline)):
            self.pipe = obj
            super().__init__(self.pipe.model,
                             self.pipe.dataloaders,
                             self.pipe.loss_fn,
                             self.pipe.writer,
                             self.pipe.KFold_class)
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
        self.sampler = sampler
        if pruner is not None:
            self.pruner = pruner
        else:
            self.pruner = MedianPruner(n_startup_trials=5,
                                       n_warmup_steps=0,
                                       interval_steps=1,
                                       n_min_trials=1)
        self.scalars_dict = dict()
        # can be changed by changing this attribute
        self.store_pickle = False

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
                   schedulers_params,
                   profiling,
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
            optimizers_params (dict):
                dictionary of the optimizers
                parameters, e.g. `{"lr": 0.001}`
            dataloaders_hyperparams (dict):
                dictionary of the dataloaders
                parameters
            models_hyperparams (dict):
                dictionary of the model
                parameters
            lr_scheduler (torch.optim):
                a learning rate scheduler
            schedulers_params (dict):
                learning rate scheduler parameters
            profiling (bool):
                whether or not you want to activate the
                profiler
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

        # for proper storing of data
        self._cross_validation = cross_validation
        self._k_folds = self.KFold_class.n_splits
        # generate optimizer
        optimizers_names = list(map(lambda x: x.__name__, optimizers))
        optimizer = eval(trial.suggest_categorical("optimizer", optimizers_names))

        # generate all the hyperparameters
        self.optimizers_param = Gridsearch._suggest_params(trial, optimizers_params)
        self.dataloaders_param = Gridsearch._suggest_params(trial, dataloaders_params)
        self.models_hyperparam = Gridsearch._suggest_params(trial, models_hyperparams)
        self.schedulers_param = Gridsearch._suggest_params(trial, schedulers_params)
        # tag for storing the results
        writer_tag += "/" + str(trial.datetime_start) # str(self.optimizers_param) + \
            #str(self.dataloaders_param) + str(self.models_hyperparam) + \
            #str(self.schedulers_param)
        # create a new model instance
        self.model = self._initialise_new_model(self.models_hyperparam)
        self.pipe = Pipeline(self.model, self.dataloaders, self.loss_fn,
                             self.writer, self.KFold_class)
        # set best_not_last
        self.pipe.best_not_last = self.best_not_last_gs
        # set the run_name
        self.pipe.run_name = str(trial.datetime_start).replace(":","-")
        loss, accuracy = self.pipe.train(optimizer, n_epochs,
                                         cross_validation,
                                         self.optimizers_param,
                                         self.dataloaders_param,
                                         lr_scheduler,
                                         self.schedulers_param,
                                         (trial, self.search_metric),
                                         profiling,
                                         parallel_tpu,
                                         keep_training,
                                         store_grad_layer_hist,
                                         n_accumulated_grads,
                                         writer_tag
                                         )
        self.scalars_dict[str(trial.datetime_start).replace(":","-")] = [self.pipe.val_loss_list_hparam,
                                                        self.pipe.val_acc_list_hparam]
        # release the run_name
        self.pipe.run_name = None
        best_loss = self.pipe.best_val_loss
        best_accuracy = self.pipe.best_val_acc
        self.writer.flush()
        # print
        self._print_output()
        
        # save model and optimizer
        save_model_and_optimizer(self.pipe.model,
                                 trial_id=str(trial.datetime_start).replace(":","-"),
                                 optimizer=self.pipe.optimizer,
                                 store_pickle=self.store_pickle)
        # returns
        if self.search_metric == "loss":
            #if self.best_not_last:
            #    self.best_val_acc_gs = max(self.best_val_acc_gs, best_accuracy)
            #    self.best_val_loss_gs = min(self.best_val_loss_gs, best_loss)
            #    return best_loss
            self.best_val_acc_gs = max(self.best_val_acc_gs, accuracy)
            self.best_val_loss_gs = min(self.best_val_loss_gs, loss)
            return loss
        else:
            #if self.best_not_last:
            #    self.best_val_acc_gs = max(self.best_val_acc_gs, best_accuracy)
            #    self.best_val_loss_gs = min(self.best_val_loss_gs, best_loss)
            #    return best_accuracy
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
              schedulers_params=None,
              profiling=False,
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
            schedulers_params (dict):
                learning rate scheduler parameters
            profiling (bool, default=False):
                whether or not you want to activate the
                profiler
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
                                             sampler=self.sampler,
                                             pruner=self.pruner,
                                             storage=self.db_url,
                                             study_name=self.study_name if self.study_name is not None else \
                                                 ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20)),
                                             load_if_exists=True)
        else:
            self.study = optuna.create_study(direction="maximize",
                                             sampler=self.sampler,
                                             pruner=self.pruner,
                                             storage=self.db_url,
                                             study_name=self.study_name if self.study_name is not None else \
                                                 ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in
                                                         range(20)),
                                             load_if_exists=True)
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
                                      schedulers_params,
                                      profiling,
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
                                schedulers_params,
                                profiling,
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
                             schedulers_params,
                             profiling,
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
                             self.bench.writer,
                             self.bench.KFold_class)
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
                                                       schedulers_params,
                                                       profiling,
                                                       parallel_tpu,
                                                       keep_training,
                                                       store_grad_layer_hist,
                                                       n_accumulated_grads,
                                                       writer_tag),
                            n_trials=self.n_trials,
                            timeout=None)

        try:
            self._results(model_name=model["name"],
                          dataset_name=dataloaders["name"])
            #save_model_and_optimizer(self.pipe.model,
            #                         model["name"] +
            #                         str(self.optimizers_param) +
            #                         str(self.dataloaders_param) +
            #                         str(self.models_hyperparam) +
            #                         str(self.schedulers_param),
            #                         self.pipe.optimizer)
        except TypeError:
            try:
                self._results(model_name=self.pipe.model.__class__.__name__,
                              dataset_name=self.pipe.dataloaders[0].dataset.__class__.__name__)
                #save_model_and_optimizer(self.pipe.model,
                #                         optimizer=self.pipe.optimizer)
            except AttributeError:
                self._results()


    def _print_output(self):
        """Printing the results of an optimisation"""
        results_string_to_print = (("\nBest Validation loss: " +
                                    str(self.pipe.best_val_loss))
                                   if self.search_metric == "loss"
                                   else ("\nBest Validation accuracy: " +
                                         str(self.pipe.best_val_acc)))

        string_to_print = ("\nModel Hyperparameters: " + str(self.models_hyperparam) +
                           "\nOptimizer: " + str(self.pipe.optimizer) +
                           "\nOptimizer parameters: " + str(self.optimizers_param) +
                           "\nDataloader parameters: " + str(self.dataloaders_param) +
                           "\nLR-scheduler parameters: " + str(self.schedulers_param) +
                           results_string_to_print
                           )
        try:
            # print models, metric and hyperparameters
            print("*" * 20 + " RESULTS " + 20 * "*"+"\n",
                  "\nModel: ", self.pipe.model.__class__.__name__,
                  string_to_print)
        except AttributeError:
            print("*" * 20 + " RESULTS " + 20 * "*"+"\n" +string_to_print)

    def _results(self, model_name="model", dataset_name="dataset"):
        """This method returns the dataframe with all the results of
        the gridsearch. It also saves the figures in the writer.

        Args:
            run_name (str):
                name of the tensorboard run
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
            print("******************** BEST TRIAL: ********************")
            trial_best = self.study.best_trial
            print("Metric Value for best trial: ", trial_best.value)
            print("Parameters Values for best trial: ", trial_best.params)
            print("DateTime start of the best trial: ", trial_best.datetime_start)
        except ValueError:
            warnings.warn("No best trial found.")
        
        for tria in trials:
            temp_list = []
            for val in tria.params.values():
                temp_list.append(val)
            if self.search_metric == "loss":
                self.list_res.append([str(tria.datetime_start).replace(":","-"), model_name,
                                      dataset_name] + temp_list + [tria.value, -1])
            else:
                self.list_res.append([str(tria.datetime_start).replace(":","-"), model_name,
                                      dataset_name] + temp_list + [np.inf, tria.value])

        self.df_res = pd.DataFrame(self.list_res, columns=["run_name", "model", "dataset"] +
                              list(trial_best.params.keys())+["loss", "accuracy"])
        # compute hyperparams correlaton
        corr, labels = self._correlation_of_hyperparams()
        if self.n_trials > 1:
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
        # average over the scalars_dict
        scalars_dict_avg = dict()
        for k, l1l2 in self.scalars_dict.items():
            scalars_dict_avg[k] = self._refactor_scalars(l1l2)
        for i in range(len(self.df_res)):
            dictio = {k:(int(v) if isinstance(v, np.int64) else v) for k,v in dict(self.df_res.iloc[i][1:-2]).items()}
            try:
                self.writer.add_hparams(dictio,
                                        {self.df_res.columns[-2]: self.df_res.iloc[i][-2],
                                         self.df_res.columns[-1]: self.df_res.iloc[i][-1]},
                                        run_name=self.df_res.iloc[i][0],
                                        scalars_lists=scalars_dict_avg[self.df_res.iloc[i][0]],
                                        best_not_last=self.best_not_last_gs)
            except KeyError:  # this happens when trials have been pruned
                pass
        
        self.writer.flush()
        
        return self.df_res

    def _refactor_scalars(self, two_lists):
        """private method to transform a list with
        many values for the same epoch into a dictionary
        compatible with ``add_scalar`` averaged per epoch

        Args:
            two_lists (list):
                two lists with pairs (value, time) with possible
                repetition of the same time

        Returns:
            list of list:
                compatible with ``add_scalar``
        """

        out0 = _inner_refactor_scalars(two_lists[0],
                                            self._cross_validation,
                                            self._k_folds)
        out1 = _inner_refactor_scalars(two_lists[1],
                                            self._cross_validation,
                                            self._k_folds)
        return out0, out1




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
        for k, v in params.items():
            if (isinstance(v, list) or isinstance(v, tuple)) and len(v)==1:
                params[k] = 2*v
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


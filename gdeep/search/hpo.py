import os
import time
import warnings
import random
from itertools import chain, combinations
from typing import Tuple, Any, Dict, Type, Optional, List, Union, Sequence, cast
import string

from typing_extensions import Literal
import torch
import optuna
from optuna.trial import TrialState
import pandas as pd
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import plotly.express as px
from optuna.pruners import MedianPruner, BasePruner
from optuna.trial._base import BaseTrial  # noqa
from optuna.study import Study
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa

from gdeep.utility import _inner_refactor_scalars, KnownWarningSilencer  # noqa
from gdeep.trainer import Trainer
from gdeep.search import Benchmark, _benchmarking_param
from gdeep.visualization import plotly2tensor
from ..utility import save_model_and_optimizer
from .hpo_config import HPOConfig
from gdeep.trainer.regularizer import Regularizer
from gdeep.utility.custom_types import Tensor, Array

SEARCH_METRICS = ("loss", "accuracy")


class GiottoSummaryWriter(SummaryWriter):
    def add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, Any],
        hparam_domain_discrete: Optional[Dict[str, List[Any]]] = None,
        run_name: Optional[str] = None,
        scalars_lists: Optional[List[List[Tuple[Any, int]]]] = None,
        best_not_last: bool = False,
    ):
        """Add a set of hyperparameters to be compared in TensorBoard.
        Args:
            hparam_dict (dict):
                Each key-value pair in the dictionary is the
                name of the hyper parameter and it's corresponding value.
                The type of the value can be one of `bool`, `string`, `float`,
                `int`, or `None`.
            metric_dict :
                Each key-value pair in the dictionary is the
                name of the metric and it's corresponding value. Note that the key used
                here should be unique in the tensorboard record. Otherwise the value
                you added by ``add_scalar`` will be displayed in hparam plugin. In most
                cases, this is unwanted.
            hparam_domain_discrete:
                A dictionary that
                contains names of the hyperparameters and all discrete values they can hold
            run_name :
                Name of the run, to be included as part of the logdir.
                If unspecified, will use current timestamp.
            scalars_lists :
                The lists for the loss and accuracy plots.
                This is a list with two lists
                (one for accuracy and one for the loss).
                Each one of the inner lists contain the
                pairs (metric_value, epoch).
            best_not_last:
                boolean flag to decide what value to store in the
                tensorboard tables for the whole training cycle
        Examples::
            from torch.utils.tensorboard import SummaryWriter
            with GiottoSummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")  # type: ignore
        if (not isinstance(hparam_dict, dict)) or (not isinstance(metric_dict, dict)):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        if not run_name:
            run_name = str(time.time()).replace(":", "-")
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)  # type: ignore
            w_hp.file_writer.add_summary(ssi)  # type: ignore
            w_hp.file_writer.add_summary(sei)  # type: ignore
            if isinstance(scalars_lists, list) or isinstance(scalars_lists, tuple):
                scalars_list_loss = scalars_lists[0]
                if best_not_last:
                    scalars_list_loss = [
                        x
                        for x in scalars_list_loss[
                            : int(np.argmin(np.array(scalars_list_loss)[:, 0])) + 1
                        ]
                    ]
                for v, t in scalars_list_loss:
                    w_hp.add_scalar("loss", v, t)
                scalars_list_acc = scalars_lists[1]
                if best_not_last:
                    scalars_list_acc = [
                        x
                        for x in scalars_list_acc[
                            : int(np.argmax(np.array(scalars_list_acc)[:, 0])) + 1
                        ]
                    ]
                for v, t in scalars_list_acc:
                    w_hp.add_scalar("accuracy", v, t)


class HyperParameterOptimization(Trainer):
    """This is the generic class that allows
    the user to perform hyperparameter search over several
    parameters such as learning rate, optimizer.
    Args:
        obj :
            either a Trainer or a Benchmark class
            instance
        search_metric :
            either ``'loss'`` or ``'accuracy'``
        n_trials :
            number of total search trials
        best_not_last :
            boolean flag that is ``True`` would use
            the best metric over epochs averaged over the folds in CV
            rather than the last value of the metrics
            over the epochs averaged over the folds
        pruner :
            Instance of an optuna pruner, can be user-defined
        sampler :
            If left unspecified, ``TPESample`` is used during single-objective
            optimization and ``NSGAIISampler`` during multi-objective optimization
        db_url:
            name of the database to connect to. For example
            ``mysql+mysqldb://usr:psw@host:port/db_name``
        study_name:
            name of the optuna study
    Examples::

        from gdeep.search import HyperParameterOptimization
        # initialise hpo, you need a `trainer`!
        search = HyperParameterOptimization(trainer, "accuracy", 2, best_not_last=True)
        # if you want to store pickle files of the models instead of the state_dicts
        search.store_pickle = True
        # dictionaries of hyperparameters
        optimizers_params = {"lr": [0.001, 0.01]}
        dataloaders_params = {"batch_size": [32, 64, 16]}
        models_hyperparams = {"n_nodes": ["200"]}
        # starting the HPO
        search.start(
            (SGD, Adam),
            3,
            False,
            optimizers_params,
            dataloaders_params,
            models_hyperparams,
            n_accumulated_grads=2,
        )
    """

    is_pipe: bool
    df_res: pd.DataFrame
    study: Study

    def __init__(
        self,
        obj: Union[Trainer, Benchmark],
        search_metric: Literal["loss", "accuracy"] = "loss",
        n_trials: int = 10,
        best_not_last: bool = False,
        pruner: Optional[BasePruner] = None,
        sampler=None,
        db_url: Optional[str] = None,
        study_name: Optional[str] = None,
    ):
        self.best_not_last_gs = best_not_last
        self.best_val_acc_gs = 0.0
        self.best_val_loss_gs = np.inf
        self.list_res: List[Any] = []
        self.db_url = db_url
        self.study_name = study_name
        if isinstance(obj, Trainer):
            self.pipe: Trainer = obj
            super().__init__(
                self.pipe.model,
                self.pipe.dataloaders,
                self.pipe.loss_fn,
                self.pipe.writer,
                self.pipe.training_metric,
                self.pipe.k_fold_class,
                regularizer=self.pipe.regularizer,
            )
            # Pipeline.__init__(self, obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            self.is_pipe = True
        elif isinstance(obj, Benchmark):
            self.bench: Benchmark = obj
            self.is_pipe = False
        self.search_metric = search_metric
        assert (
            self.search_metric in SEARCH_METRICS
        ), "Wrong search_metric! Either `loss` or `accuracy`"
        self.n_trials = n_trials
        self.val_epoch = 0
        self.train_epoch = 0
        self.sampler = sampler
        if pruner is not None:
            self.pruner = pruner
        else:
            self.pruner = MedianPruner(
                n_startup_trials=5, n_warmup_steps=0, interval_steps=1, n_min_trials=1
            )
        self.scalars_dict: Dict[str, Any] = dict()
        # can be changed by changing this attribute
        self.store_pickle: bool = False

    def _initialise_new_model(
        self, models_hyperparam: Optional[Dict[str, Any]]
    ) -> torch.nn.Module:
        """private method to find the maximal compatible set
        between models and hyperparameters

        Args:
            models_hyperparam (dict):
                model selected hyperparameters

        Returns:
            nn.Module
                torch nn.Module
        """
        new_model = None
        if models_hyperparam:
            list_of_params_keys = HyperParameterOptimization._powerset(
                list(models_hyperparam.keys())
            )
            list_of_params_keys.reverse()
            for params_keys in list_of_params_keys:
                sub_models_hyperparam = {
                    k: models_hyperparam[k]
                    for k in models_hyperparam.keys()
                    if k in params_keys
                }
                try:
                    # print(sub_models_hyperparam)
                    new_model = type(self.model)(**sub_models_hyperparam)  # noqa
                    # print(new_model.state_dict())
                    raise ValueError
                except TypeError:  # when the parameters do not match the model
                    pass
                except ValueError:  # when the parameters match the model
                    break
        else:
            new_model = type(self.model)()

        if new_model is not None:
            warnings.warn("Model cannot be re-initialized. Using existing one.")
            new_model = self.model
        assert (
            new_model is not None
        ), "There is a problem with the re-initialization of the model"
        return new_model

    def _objective(self, trial: BaseTrial, config: HPOConfig):
        """default callback function for optuna's study

        Args:
            trial:
                the independent variable
            config:
                configuration class HPOConfig,
                containing all the parameters needed
        Returns:
            float
                metric (either loss or accuracy)
        """

        # for proper storing of data
        self._cross_validation = config.cross_validation
        self._k_folds = self.k_fold_class.n_splits  # type: ignore
        # generate optimizer
        optimizer = HyperParameterOptimization._new_suggest_categorical(
            trial, "optimizer", config.optimizers
        )

        # generate all the hyperparameters
        self.optimizers_param = HyperParameterOptimization._suggest_params(
            trial, config.optimizers_params
        )
        self.dataloaders_param = HyperParameterOptimization._suggest_params(
            trial, config.dataloaders_params
        )
        self.models_hyperparam = HyperParameterOptimization._suggest_params(
            trial, config.models_hyperparams
        )
        self.schedulers_param = HyperParameterOptimization._suggest_params(
            trial, config.schedulers_params
        )
        self.regularization_param = HyperParameterOptimization._suggest_params(
            trial, config.regularization_params
        )
        # tag for storing the results
        config.writer_tag += "/" + str(
            trial.datetime_start
        )  # str(self.optimizers_param) + \
        # str(self.dataloaders_param) + str(self.models_hyperparam) + \
        # str(self.schedulers_param)
        # create a new model instance
        self.model = self._initialise_new_model(self.models_hyperparam)
        if self.regularization_param is not None:
            if "regularizer" in self.regularization_param:
                reg = self.regularization_param.pop("regularizer")(
                    **self.regularization_param
                )
                self.pipe = Trainer(
                    self.model,
                    self.dataloaders,
                    self.loss_fn,
                    self.writer,
                    self.training_metric,
                    self.k_fold_class,
                    regularizer=reg,
                )
        else:
            self.pipe = Trainer(
                self.model,
                self.dataloaders,
                self.loss_fn,
                self.writer,
                self.training_metric,
                self.k_fold_class,
            )
        # set best_not_last
        self.pipe.best_not_last = self.best_not_last_gs
        # set the run_name
        self.pipe.run_name = str(trial.datetime_start).replace(":", "-")
        loss, accuracy = self.pipe.train(
            optimizer,
            config.n_epochs,
            config.cross_validation,
            self.optimizers_param,
            self.dataloaders_param,
            config.lr_scheduler,
            self.schedulers_param,
            (trial, self.search_metric),
            config.profiling,
            config.parallel_tpu,
            config.keep_training,
            config.store_grad_layer_hist,
            config.n_accumulated_grads,
            config.writer_tag,
        )

        # dict of the loss per epoch of the training here above. If CV, multiple scalars are stored
        scalars_dict_value = [
            self.pipe.val_loss_list_hparam,
            self.pipe.val_acc_list_hparam,
        ]
        scalar_dict_key = str(trial.datetime_start).replace(":", "-")
        self.scalars_dict[scalar_dict_key] = scalars_dict_value
        # release the run_name
        self.pipe.run_name = None
        self.writer.flush()  # type: ignore
        # print
        self._print_output()

        # save model and optimizer
        save_model_and_optimizer(
            self.pipe.model,
            trial_id=str(trial.datetime_start).replace(":", "-"),
            optimizer=self.pipe.optimizer,
            store_pickle=self.store_pickle,
        )

        # save results to tensorboard
        model_name, dataset_name = self._extract_model_and_dataset_name(
            config.writer_tag
        )

        self._store_trial_to_tb(
            trial,
            scalars_dict_value,
            scalar_dict_key,
            model_name,
            dataset_name,
            loss,
            accuracy,
        )

        # returns
        if self.search_metric == "loss":
            self.best_val_acc_gs = max(self.best_val_acc_gs, accuracy)
            self.best_val_loss_gs = min(self.best_val_loss_gs, loss)
            return loss
        else:
            self.best_val_acc_gs = max(self.best_val_acc_gs, accuracy)
            self.best_val_loss_gs = min(self.best_val_loss_gs, loss)
            return accuracy

    def _extract_model_and_dataset_name(self, writer_tag: str) -> Tuple[str, str]:
        """Extract the model and dataset from the writer_tag"""
        index_ds = writer_tag.find("Dataset:")
        if index_ds == -1:
            dataset_name = self.pipe.dataloaders[0].dataset.__class__.__name__
        else:
            dataset_name = writer_tag[index_ds + 8 : writer_tag.find("|Model:")]
        index_md = writer_tag.find("|Model:")
        if index_md == -1:
            model_name = self.pipe.model.__class__.__name__
        else:
            model_name = writer_tag[index_md + 7 : writer_tag.find("/")]
        return model_name, dataset_name

    def start(
        self,
        optimizers: List[Type[Optimizer]],
        n_epochs: int = 1,
        cross_validation: bool = False,
        optimizers_params: Optional[Dict[str, Any]] = None,
        dataloaders_params: Optional[Dict[str, Any]] = None,
        models_hyperparams: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        schedulers_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
        profiling: bool = False,
        parallel_tpu: bool = False,
        keep_training: bool = False,
        store_grad_layer_hist: bool = False,
        n_accumulated_grads: int = 0,
        writer_tag: str = "",
    ) -> None:
        """method to be called when starting the hyperparameter optimization
        Args:
            optimizers:
                list of torch optimizers classes, not instances
            n_epochs:
                number of training epochs
            cross_validation:
                whether or not to use cross-validation
            optimizers_params:
                dictionary of optimizers params
            dataloaders_params:
                dictionary of dataloaders parameters
            models_hyperparams:
                dictionary of model parameters
            lr_scheduler:
                torch learning rate scheduler class
            schedulers_params:
                learning rate scheduler parameters
            profiling :
                whether or not you want to activate the
                profiler
            parallel_tpu:
                boolean value to run the computations
                on multiple TPUs
            n_accumulated_grads (int, default=0):
                number of accumulated gradients. It is
                considered only if a positive integer
            keep_training:
                bool flag to decide whether to continue
                training or not
            store_grad_layer_hist:
                flag to store the gradients of the layers in the
                tensorboard histograms
            writer_tag:
                tag to prepend to the output
                on tensorboard
        """
        if self.search_metric == "loss":
            self.study = optuna.create_study(
                direction="minimize",
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.db_url,
                study_name=self.study_name
                if self.study_name is not None
                else "".join(
                    random.choice(string.ascii_uppercase + string.digits)
                    for _ in range(20)
                ),
                load_if_exists=True,
            )
        else:
            self.study = optuna.create_study(
                direction="maximize",
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.db_url,
                study_name=self.study_name
                if self.study_name is not None
                else "".join(
                    random.choice(string.ascii_uppercase + string.digits)
                    for _ in range(20)
                ),
                load_if_exists=True,
            )
        config = HPOConfig(
            optimizers,
            n_epochs,
            cross_validation,
            optimizers_params,
            dataloaders_params,
            models_hyperparams,
            lr_scheduler,
            schedulers_params,
            regularization_params,
            profiling,
            parallel_tpu,
            keep_training,
            store_grad_layer_hist,
            n_accumulated_grads,
            writer_tag,
        )

        if self.is_pipe:
            # in the __init__, self.model and self.dataloaders are
            # already initialized. So they exist also in _objective()
            self._inner_optimization_fun(
                self.model, self.dataloaders, self.regularizer, config
            )

        else:
            _benchmarking_param(
                self._inner_optimization_fun,
                (
                    self.bench.models_dicts,
                    self.bench.dataloaders_dicts,
                    self.bench.regularizers_dicts,
                ),
                config,
            )
        # self._store_to_tensorboard()

    def _inner_optimization_fun(
        self,
        model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        dataloaders: Union[
            List[DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]]],
            Dict[str, DataLoader],
        ],
        regularizers: Union["Regularizer", None, Dict[str, Union["Regularizer", None]]],
        config: HPOConfig,
    ) -> None:
        """private method to be decorated with the
        benchmark decorator to have benchmarking
        or simply used as is if no benchmarking is
        needed
        """

        try:
            config.writer_tag = (
                "Dataset:" + dataloaders["name"] + "|Model:" + model["name"]  # type: ignore
            )
            super().__init__(
                model["model"],  # type: ignore
                dataloaders["dataloaders"],  # type: ignore
                self.bench.loss_fn,
                self.bench.writer,
                self.bench.training_metric,
                self.bench.k_fold_class,
                # regularizers['regularizer']
            )
        except TypeError:
            pass

        self.study.optimize(  # type: ignore
            lambda tr: self._objective(tr, config), n_trials=self.n_trials, timeout=None
        )

        try:
            self._results(model_name=model["name"], dataset_name=dataloaders["name"])  # type: ignore
            # save_model_and_optimizer(self.pipe.model,
            #                         model["name"] +
            #                         str(self.optimizers_param) +
            #                         str(self.dataloaders_param) +
            #                         str(self.models_hyperparam) +
            #                         str(self.schedulers_param),
            #                         self.pipe.optimizer)
        except TypeError:
            try:
                self._results(
                    model_name=self.pipe.model.__class__.__name__,
                    dataset_name=self.pipe.dataloaders[0].dataset.__class__.__name__,
                )
                # save_model_and_optimizer(self.pipe.model,
                #                         optimizer=self.pipe.optimizer)
            except AttributeError:
                self._results()

    def _print_output(self) -> None:
        """Printing the results of an optimization"""
        results_string_to_print = (
            ("\nBest Validation loss: " + str(self.pipe.best_val_loss))
            if self.search_metric == "loss"
            else ("\nBest Validation accuracy: " + str(self.pipe.best_val_acc))
        )

        string_to_print = (
            "\nModel Hyperparameters: "
            + str(self.models_hyperparam)
            + "\nOptimizer: "
            + str(self.pipe.optimizer)
            + "\nOptimizer parameters: "
            + str(self.optimizers_param)
            + "\nDataloader parameters: "
            + str(self.dataloaders_param)
            + "\nLR-scheduler parameters: "
            + str(self.schedulers_param)
            + "\nRegularizer parameters: "
            + str(self.regularization_param)
            + results_string_to_print
        )
        try:
            # print models, metric and hyperparameters
            print(
                "*" * 20 + " RESULTS " + 20 * "*" + "\n",
                "\nModel: ",
                self.pipe.model.__class__.__name__,
                string_to_print,
            )
        except AttributeError:
            print("*" * 20 + " RESULTS " + 20 * "*" + "\n" + string_to_print)

    def _store_to_list_each_step(
        self,
        trial: BaseTrial,
        model_name: str,
        dataset_name: str,
        loss: Optional[float] = np.inf,
        accuracy: Optional[float] = -1.0,
        list_res: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Private method to store all the HPO parameters
        of one trial

        Args:
            trial (optuna.trial):
                the trial at hand
            model_name (str):
                name of the model
            dataset_name (str):
                name of the dataset
            loss (float, default np.inf)
                the value of the loss for the current trial
            accuracy (float, default -1):
                the value of the accuracy for the current trial
            list_res (list):
                list of results
        Returns:
            list:
                list of HPOs and metrics. the first element is the
                run name and the last two are the metrics (loss
                and accuracy)
        """
        if list_res is None:
            list_res = []
        temp_list = []
        for val in trial.params.values():
            temp_list.append(val)
        if self.search_metric == "loss":
            try:
                loss = trial.value  # type: ignore
            except AttributeError:
                loss = loss
            list_res.append(
                [str(trial.datetime_start).replace(":", "-"), model_name, dataset_name]
                + temp_list
                + [loss, -1]  # type: ignore
            )
        else:
            try:
                accuracy = trial.value  # type: ignore
            except AttributeError:
                accuracy = accuracy
            list_res.append(
                [str(trial.datetime_start).replace(":", "-"), model_name, dataset_name]
                + temp_list
                + [np.inf, accuracy]  # type: ignore
            )
        return list_res

    def _store_trial_to_tb(
        self,
        trial: BaseTrial,
        scalars_dict_value: List[List[List[float]]],
        scalar_dict_key: str,
        model_name: str,
        dataset_name: str,
        loss: float,
        accuracy: float,
    ) -> None:
        """store hyperparameters of a single trial to
        tensorboard
        """
        list_res = self._store_to_list_each_step(
            trial, model_name, dataset_name, loss, accuracy
        )

        keys = ["model", "dataset"] + list(trial.params.keys())

        # average over the scalars_dict
        scalars_dict_avg = self._refactor_scalars(scalars_dict_value)
        # dict of parameters
        dictionary_ = {
            k: (int(v) if isinstance(v, np.int64) or isinstance(v, np.int32) else v)  # type: ignore
            for k, v in zip(keys, list_res[0][1:-2])
        }

        try:
            self.writer.add_hparams(  # type: ignore
                dictionary_,
                {"loss": list_res[0][-2], "accuracy": list_res[0][-1]},
                run_name=scalar_dict_key,
                scalars_lists=scalars_dict_avg,  # type: ignore
                best_not_last=self.best_not_last_gs,  # type: ignore
            )
        except KeyError:  # this happens when trials have been pruned
            pass

        self.writer.flush()  # type: ignore

    def _results(
        self, model_name: str = "model", dataset_name: str = "dataset"
    ) -> pd.DataFrame:
        """This method returns the dataframe with all the results of
        the hyperparameters optimization.
        It also saves the figures in the writer.
        Args:
            model_name:
                the model name for the
                tensorboard hpo table
            dataset_name:
                the dataset name for the
                tensorboard hpo table
        Returns:
            pd.DataFrame:
                the hyperparameter table
        """
        self.list_res = []
        trials = self.study.trials
        pruned_trials = self.study.get_trials(
            deepcopy=False, states=(TrialState.PRUNED,)
        )
        complete_trials = self.study.get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,)
        )

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
            warnings.warn("No best trial found. Using the first trial")
            trial_best = self.study.trials[0]

        for trial in trials:
            self._store_to_list_each_step(
                trial, model_name, dataset_name, trial.value, trial.value, self.list_res
            )

        self.df_res = pd.DataFrame(
            self.list_res,
            columns=["run_name", "model", "dataset"]
            + list(trial_best.params.keys())
            + ["loss", "accuracy"],
        )
        # compute hyperparams correlation
        corr, labels = self._correlation_of_hyperparams()
        if self.n_trials > 1:
            try:
                fig2 = px.imshow(
                    corr,
                    labels=dict(x="Parameters", y="Parameters", color="Correlation"),
                    x=labels,
                    y=labels,
                )
                fig2.update_xaxes(side="top")
                # fig2.show()
                img2 = plotly2tensor(fig2)

                self.writer.add_images(  # type: ignore
                    "HPO correlation: " + model_name + " " + dataset_name,
                    img2,
                    dataformats="HWC",
                )
                self.writer.flush()  # type: ignore
            except ValueError:
                warnings.warn(
                    "Cannot send the picture of the correlation"
                    + " matrix to tensorboard"
                )

        return self.df_res

    def _correlation_of_hyperparams(self) -> Tuple[Array, List[Any]]:
        """Correlations of numerical hyperparameters"""
        list_of_arrays = []
        labels = []
        for col in self.df_res.columns:
            vals = self.df_res[col].values
            # print("type here", vals.dtype)
            if vals.dtype in [
                np.float16,
                np.float64,
                np.float32,
                np.int32,
                np.int64,
                np.int16,
            ]:
                list_of_arrays.append(vals)
                labels.append(col)
        # print(list_of_arrays)
        with KnownWarningSilencer():
            corr = np.corrcoef(np.array(list_of_arrays))
        return corr, labels

    def _refactor_scalars(
        self, two_lists: List[List[List[float]]]
    ) -> Tuple[List[Any], List[Any]]:
        """private method to transform a list with
        many values for the same epoch into a dictionary
        compatible with ``add_scalar`` averaged per epoch
        Args:
            two_lists :
                two lists with pairs (value, time) with possible
                repetition of the same time
        Returns:
            list of list:
                compatible with ``add_scalar``
        """

        out0 = _inner_refactor_scalars(
            two_lists[0], self._cross_validation, self._k_folds
        )
        out1 = _inner_refactor_scalars(
            two_lists[1], self._cross_validation, self._k_folds
        )
        return out0, out1

    @staticmethod
    def _suggest_params(
        trial: BaseTrial, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Utility function to generate the parameters
        for the hyperparameter search. It is based on optuna `suggest_<type>`.
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
            if (isinstance(v, list) or isinstance(v, tuple)) and len(v) == 1:
                params[k] = 2 * v
        # param_temp = {}
        param_temp = {
            k: HyperParameterOptimization._new_suggest_float(trial, k, *v)
            for k, v in params.items()
            if (isinstance(v, list) or isinstance(v, tuple))
            and (isinstance(v[0], float) or isinstance(v[1], float))
        }
        param_temp2 = {
            k: trial.suggest_int(k, *v)
            for k, v in params.items()
            if (isinstance(v, list) or isinstance(v, tuple))
            and (isinstance(v[0], int) or isinstance(v[1], int))
        }
        param_temp.update(param_temp2)
        param = {
            k: HyperParameterOptimization._new_suggest_categorical(trial, k, v)
            for k, v in params.items()
            if (isinstance(v, list) or isinstance(v, tuple))
            and not (isinstance(v[0], int) or isinstance(v[1], int))
            and not (isinstance(v[0], float) or isinstance(v[1], float))
        }
        param.update(param_temp)
        # print("suggested parameters:", param)
        return param

    @staticmethod
    def _new_suggest_categorical(trial, name: str, choices: Sequence) -> Any:
        """A modification of the Optuna function, in order to remove the
        constraints on the type for categorical choices.
        Expected to get a list with items of the same type (or base class).
        For example, ``choices`` being a list of Callables or classes would
        work.
        """

        if isinstance(choices, list) or isinstance(choices, tuple):
            if (
                isinstance(choices[0], str)
                or isinstance(choices[0], int)
                or isinstance(choices[0], float)
            ):
                return trial.suggest_categorical(name, choices)
            else:  # in case optuna cannot handle the types
                dict_choices = {x.__name__: x for x in choices}
                key = trial.suggest_categorical(
                    name, dict_choices.keys()
                )  # random choice on the names
                return dict_choices[key]  # return the corresponding value
        return random.choice(choices)

    @staticmethod
    def _new_suggest_float(trial, name, low, high, step=None, log=False):
        """A modification of the Optuna function, in order to remove the `*`"""
        return trial.suggest_float(name, low, high, log=log, step=step)

    @staticmethod
    def _powerset(iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

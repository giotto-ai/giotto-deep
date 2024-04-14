import os
import copy
import time
from functools import wraps
import warnings
from typing import (
    Tuple,
    Optional,
    Callable,
    Any,
    Dict,
    List,
    Type,
    Union,
    TYPE_CHECKING,
)

import torch.nn.functional as f
from torch.optim import Optimizer
import torch
from optuna.trial._base import BaseTrial
import numpy as np
from tqdm import tqdm
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import optuna
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler  # noqa

from ..utility.optimization import MissingClosureError
from gdeep.models import ModelExtractor
from gdeep.utility import _inner_refactor_scalars
from gdeep.utility import DEVICE


from gdeep.trainer.regularizer import Regularizer
#if TYPE_CHECKING:
#    from gdeep.trainer.regularizer import Regularizer
from .metrics import accuracy

from gdeep.utility.custom_types import Tensor

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
    import torch_xla.distributed.parallel_loader as pl  # type: ignore

    try:
        DEVICE = xm.xla_device()
    except NameError:
        pass
    print("Using TPU!")
except ModuleNotFoundError:
    pass


def _add_data_to_tb(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """decorator to store PR data to tensorboard"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        pred, val_loss, correct = func(*args, **kwargs)
        try:
            # add data to tensorboard
            Trainer._add_pr_curve_tb(
                torch.vstack(pred),
                kwargs["class_label"],
                kwargs["class_probs"],
                kwargs["writer_tag"] + "/validation",
            )
        except NotImplementedError:
            warnings.warn("The PR curve is not being filled because too few data exist")
        return pred, val_loss, correct

    return wrapper


class Trainer:
    """This is the generic class that allows
    the user to benchmark models over architectures
    datasets, regularisation, metrics... in one line
    of code.

    Args:
        model :
            standard torch model
        dataloaders (list of utils.DataLoader):
            list of standard torch DataLoaders, e.g.
            `[dl_tr, dl_val, dl_ts]`
        loss_fn :
            loss function to average over batches
        writer :
            tensorboard writer
        training_metric:
            the function that computes the metric: it shall
            have two arguments, one for the prediction
            and the other for the ground truth
        k_fold_class (sklearn.model_selection, default ``KFold(5, shuffle=True)``):
            the class instance to implement the KFold, can be
            any of the Splitter classes of sklearn. More
            info at https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
        print_every:
            The number of training steps performed between each information printout
        regularizer:
            a gdeep regularizer

    Examples::

        from torch import nn
        from torch.optim import SGD
        from sklearn.model_selection import StratifiedKFold
        from gdeep.models import FFNet
        from gdeep.trainer import Trainer
        from gdeep.data import BuildDatasets, BuildDataLoaders
        from gdeep.search import GiottoSummaryWriter
        # model
        class model1(nn.Module):
            def __init__(self):
                super(model1, self).__init__()
                self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=[3, 5, 10, 5, 2]))
            def forward(self, x):
                return self.seqmodel(x)

        model = model1()
        # dataloaders
        bd = BuildDatasets(name="DoubleTori")
        ds_tr, ds_val, _ = bd.build_datasets()
        dl = BuildDataLoaders((ds_tr, ds_val))
        dl_tr, dl_val, dl_ts = dl.build_dataloaders(batch_size=23)

        # loss function
        loss_fn = nn.CrossEntropyLoss()
        # tb writer
        writer = GiottoSummaryWriter()
        # pipeline
        pipe = Trainer(model, [dl_tr, dl_val, dl_ts],
                        loss_fn, writer, None,
                        StratifiedKFold(5, shuffle=True))
        # then one needs to train the model using the pipeline!
        pipe.train(SGD, 2, True, {"lr": 0.001}, n_accumulated_grads=5)

    """

    scheduler: Optional[_LRScheduler]
    writer: Optional[SummaryWriter]
    registered_hook: Optional[
        Callable[[int, Optimizer, ModelExtractor, Optional[SummaryWriter]], Any]
    ] = None
    regularizer: Optional[Regularizer] = None

    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: List[DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]]],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        writer: Optional[SummaryWriter] = None,
        training_metric: Optional[Callable[[Tensor, Tensor], float]] = None,
        k_fold_class: Optional[BaseCrossValidator] = None,
        print_every: int = 1,
        regularizer: Optional["Regularizer"] = None,
    ) -> None:
        self.print_every = print_every if print_every > 0 else 1
        self.model = model
        self.initial_model = copy.deepcopy(self.model)
        assert 0 < len(dataloaders) < 4, "Length of dataloaders must be 1, 2, or 3"
        self.dataloaders = dataloaders  # train and test
        self.train_epoch = 0
        self.val_epoch = 0
        self.best_val_loss = np.inf
        self.best_val_acc = 0.0
        self.loss_fn = loss_fn
        if training_metric:
            self.training_metric = training_metric
        else:
            self.training_metric = accuracy
        # integrate tensorboard
        self.writer = writer
        if self.writer is None:
            warnings.warn("No writer detected")
        self.check_has_trained: bool = False
        # used in gradient clipping
        self.clip: int = 5
        # used by hpo:
        self.run_name: Optional[str] = None
        self.val_loss_list_hparam: List[List[float]] = []
        self.val_acc_list_hparam: List[List[float]] = []
        self.best_not_last: bool = False
        # profiler
        self.prof: Any = None
        self.regularizer = regularizer
        # self.train_loss_fn = loss_fn
        # if self.regularizer:
        # def tmploss(X,y):
        #    return loss_fn(X,y) + self.regularizer.regularization_penalty(self.model)
        # self.train_loss_fn = tmploss
        if not k_fold_class:
            self.k_fold_class = KFold(5, shuffle=True)
        else:
            self.k_fold_class = k_fold_class

    def _set_initial_model(self) -> None:
        """This private method is used to set
        the initial_model"""
        self.initial_model = copy.deepcopy(self.model)

    def _reset_model(self) -> None:
        """Private method to reset the initial model weights.
        This function is essential for the cross-validation
        procedure.
        """
        self.model = copy.deepcopy(self.initial_model)

    def _optimisation_step(
        self,
        steps: int,
        loss: Tensor,
        epoch_loss: float,
        batch_metric: float,
        batch: int,
        closure: Callable[[], Tensor],
    ) -> float:
        """Backpropagation"""
        if self.n_accumulated_grads < 2:  # usual case for stochastic gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            if DEVICE.type == "xla":
                xm.optimizer_step(
                    self.optimizer, barrier=True
                )  # Note: Cloud TPU-specific code!
            else:
                try:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # type: ignore
                    self.optimizer.step()
                except (MissingClosureError,):
                    self.optimizer.step(closure)  # type: ignore

        else:  # if we use gradient accumulation techniques
            (loss / self.n_accumulated_grads).backward()
            if (
                batch + 1
            ) % self.n_accumulated_grads == 0:  # do the optimization step only after the accumulations
                if DEVICE.type == "xla":
                    xm.optimizer_step(
                        self.optimizer, barrier=True
                    )  # Note: Cloud TPU-specific code!
                else:
                    try:
                        torch.nn.utils.clip_grad_norm_(  # type: ignore
                            self.model.parameters(), self.clip
                        )
                        self.optimizer.step()
                    except (MissingClosureError,):
                        self.optimizer.step(closure)  # type: ignore
                self.optimizer.zero_grad()
        if batch % self.print_every == 0:
            epoch_loss += loss.item()
            print(
                f"Batch training loss:  {epoch_loss / (batch + 1)}",
                f" \tBatch training {self.training_metric.__name__}: ",
                batch_metric,
                " \t[",
                batch + 1,
                "/",
                steps,
                "]                     ",
                end="\r",
            )
        return epoch_loss

    def _send_to_device(
        self, x: Union[Tensor, List[Tensor]], y: Tensor
    ) -> Tuple[Tensor, Union[Tensor, List[Tensor]], Tensor]:
        """use this private method to send the
        ``x`` and ``y`` to the ``DEVICE``.

        Args:
            x:
                the input of the model, either a List[Tensor] or a Tensor
            y:
                the label

        Returns:
            (Tensor, Union[Tensor, List[Tensor]], Tensor)
                the prediction for x, x and the label

        """
        new_x: List[Tensor] = []
        if isinstance(x, tuple) or isinstance(x, list):
            for xi in x:
                new_x.append(xi.to(DEVICE))
            x = new_x
            prediction = self.model(*x)
            if hasattr(prediction, "logits"):  # unwrapper for HuggingFace BERT model
                prediction = prediction.logits  # unwrapper for HuggingFace BERT model
        else:
            x = x.to(DEVICE)
            prediction = self.model(x)
            if hasattr(prediction, "logits"):  # unwrapper for HuggingFace BERT model
                prediction = prediction.logits  # unwrapper for HuggingFace BERT model
        y = y.to(DEVICE)

        return prediction, x, y

    def _inner_train_loop(
        self,
        dl_tr: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        writer_tag: str,
        size: int,
        steps: int,
    ) -> Tuple[float, float]:
        """Private method to run the loop
        over the batches for the optimisation"""

        if self.prof is not None:
            self.prof.start()
        metric_list = []
        epoch_loss = 0.0
        for batch, (X, y) in enumerate(dl_tr):

            def closure() -> Tensor:
                loss2 = self.loss_fn(self.model(X), y)
                loss2.backward()
                return loss2

            pred, X, y = self._send_to_device(X, y)
            batch_metric = self.training_metric(pred, y)
            metric_list.append(batch_metric)
            loss = self.loss_fn(pred,y)
            if self.regularizer is not None:
                penalty = self.regularizer.regularization_penalty(self.model)
                loss += penalty
                
            # Save to tensorboard
            try:
                self.writer.add_scalar(  # type: ignore
                    writer_tag + "/loss/train",
                    loss.item(),
                    self.train_epoch * len(dl_tr) + batch,
                )

                try:
                    top2_pred = torch.topk(pred, 2, -1).values
                    try:
                        self.writer.add_histogram(  # type: ignore
                            writer_tag + "/predictions/train",
                            torch.abs(torch.diff(top2_pred, dim=-1)),
                            self.train_epoch * steps + batch,
                        )
                    except ValueError:
                        warnings.warn(
                            f"The histogram is empty, most likely because your loss"
                            f" is exploding. Try use gradient clipping."
                        )
                except RuntimeError:
                    pass
            except AttributeError:
                pass
            epoch_loss = self._optimisation_step(
                steps, loss, epoch_loss, batch_metric, batch, closure
            )

            if self.prof is not None:
                self.prof.step()

        if self.prof is not None:
            self.prof.stop()

        # epoch metric and loss:
        epoch_metric = sum(metric_list) / len(metric_list)
        epoch_loss /= steps
        return epoch_metric, epoch_loss

    def _train_loop(
        self,
        dl_tr: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        writer_tag: str = "",
    ) -> Tuple[float, float]:
        """private method to run a single training
        loop
        """
        self.model = self.model.to(DEVICE)
        self.model.train()
        try:
            length: int = len(dl_tr.sampler.indices)  # type: ignore
        except AttributeError:
            length: int = len(dl_tr.dataset)  # type: ignore
        steps = len(dl_tr)
        tik = time.time()
        assert self.n_accumulated_grads <= steps, (
            "The number of" + " accumulated gradients shall be diminished!"
        )
        epoch_metric, epoch_loss = self._inner_train_loop(
            dl_tr, writer_tag, length, steps
        )
        print(
            f"Epoch training loss: {epoch_loss:>8f} \tEpoch training "
            f"{self.training_metric.__name__}: {epoch_metric:.2f}% ".ljust(100)
        )
        try:
            self.writer.flush()  # type: ignore
        except AttributeError:
            pass
        print(f"Time taken for this epoch: {round(time.time() - tik):.2f}s")
        try:
            print(f"Learning rate value: {self.optimizer.param_groups[0]['lr']:0.8f}")
        except KeyError:
            pass
        return epoch_metric, epoch_loss

    def _val_loop(
        self,
        dl_val: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        writer_tag: str = "",
    ) -> Tuple[float, float]:
        """private method to run a single validation
        loop
        """
        self.model = self.model.to(DEVICE)
        try:
            size = len(dl_val.sampler.indices)  # type: ignore
        except AttributeError:
            size = len(dl_val.dataset)  # type: ignore
        class_label: List[Tensor] = []
        class_probs: List[List[Tensor]] = []
        self.model.eval()

        pred_list, epoch_loss, epoch_metric = self._inner_loop(  # type: ignore
            dl=dl_val,  # type: ignore
            class_probs=class_probs,  # type: ignore
            class_label=class_label,  # type: ignore
            writer_tag=writer_tag,  # type: ignore
        )
        # accuracy
        try:
            if not self.run_name:
                self.writer.add_scalar(  # type: ignore
                    writer_tag + "/metric/validation", epoch_metric, self.val_epoch
                )
            else:
                self.val_acc_list_hparam.append([epoch_metric, self.val_epoch])
                self.val_loss_list_hparam.append([epoch_loss, self.val_epoch])

            try:
                top2_pred = torch.topk(torch.vstack(pred_list), 2, -1).values
                try:
                    self.writer.add_histogram(  # type: ignore
                        writer_tag + "/predictions/validation",
                        torch.abs(torch.diff(top2_pred, dim=-1)),
                        self.val_epoch,
                    )
                except ValueError:
                    warnings.warn(
                        f"The histogram is empty, most likely because your loss"
                        f" is exploding. Try use gradient clipping."
                    )
            except RuntimeError:
                pass
        except AttributeError:
            pass
        print(
            f"Validation results: \n {self.training_metric.__name__}: {epoch_metric:.2f}%, \
                Avg loss: {epoch_loss:>8f} \n"
        )
        try:
            self.writer.flush()  # type: ignore
        except AttributeError:
            pass

        return epoch_loss, epoch_metric

    @staticmethod
    def _add_pr_curve_tb(
        pred: Tensor,
        class_label: List[Tensor],
        class_probs: List[List[Tensor]],
        writer: SummaryWriter,
        writer_tag: str = "",
    ) -> None:
        """private function to add the PR curve
        to tensorboard"""
        probs = torch.cat([torch.stack(batch) for batch in class_probs]).cpu()
        labels = torch.cat(class_label).cpu()
        for class_index in range(len(pred[0])):
            tensorboard_truth = 1 * (labels == class_index).flatten()
            tensorboard_probs = probs[:, class_index]
            # print(tensorboard_truth)
            # print(tensorboard_probs)
            try:
                writer.add_pr_curve(
                    writer_tag + "/class = " + str(class_index),
                    tensorboard_truth,
                    tensorboard_probs,
                    global_step=0,
                )
            except AttributeError:
                warnings.warn("Cannot store data in the PR curve")
            except ValueError:
                warnings.warn("Cannot store data in the PR curve")

    @_add_data_to_tb  # type: ignore
    def _inner_loop(
        self,
        *,
        dl: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        class_probs: List[List[Tensor]],
        class_label: List[Tensor],
        writer_tag: str,  # noqa
    ) -> Tuple[List[Tensor], float, float]:
        """private function used inside the test
        and validation loops"""
        pred_list = []
        batch_metric_list = []
        loss = 0.0
        with torch.no_grad():
            for X, y in dl:
                pred, X, y = self._send_to_device(X, y)
                pred_list.append(pred)
                class_probs_batch = [f.softmax(el, dim=0) for el in pred]
                class_probs.append(class_probs_batch)
                loss += self.loss_fn(pred, y).item()
                batch_metric = self.training_metric(pred, y)
                batch_metric_list.append(batch_metric)
                class_label.append(y)
        epoch_metric = sum(batch_metric_list) / len(batch_metric_list)
        epoch_loss = loss / len(batch_metric_list)
        return pred_list, epoch_loss, epoch_metric

    def _init_profiler(
        self, profiling: bool, cross_validation: bool, n_epochs: int, k_folds: int
    ) -> None:
        """initialise the profler for profiling"""
        # profiling
        active: int = 10
        if not cross_validation:
            active = n_epochs - 2
        else:
            active = k_folds * (n_epochs - 2)

        if profiling:
            try:
                self.prof = torch.profiler.profile(  # type: ignore
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,  # type: ignore
                        torch.profiler.ProfilerActivity.CUDA,  # type: ignore
                    ],
                    schedule=torch.profiler.schedule(  # type: ignore
                        wait=1, warmup=1, active=active, repeat=1
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(  # type: ignore
                        os.path.join(
                            ".",
                            "runs",
                            (
                                self.model.__class__.__name__ + str(datetime.today())
                            ).replace(":", "-"),
                        ),
                        worker_name="worker",
                    ),
                    record_shapes=True,
                    profile_memory=True
                    # with_stack=True
                )
            except AssertionError:
                pass

    def _init_optimizer_and_scheduler(
        self,
        keep_training: bool,
        cross_validation: bool,
        optimizer: Type[Optimizer],
        optimizers_param: Dict[str, Any],
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset or maintain the LR scheduler and the
        optimizer depending on the training"""
        if not (self.check_has_trained and keep_training):
            # reset the model weights
            self._reset_model()
            self.optimizer = optimizer(self.model.parameters(), **optimizers_param)
            if lr_scheduler is not None:
                if scheduler_params:
                    self.scheduler = lr_scheduler(self.optimizer, **scheduler_params)
                else:
                    self.scheduler = lr_scheduler(self.optimizer)
        elif cross_validation:
            # reset the model weights
            self._reset_model()
            # do not re-initialise the optimizer if keep-training
            dict_param = self.optimizer.param_groups[0]
            dict_param.pop("params", None)  # model.parameters()
            dict_param.pop("initial_lr", None)
            self.optimizer.__init__(self.model.parameters(), **dict_param)  # type: ignore
            if lr_scheduler is not None:  # reset scheduler
                if scheduler_params:
                    self.scheduler = lr_scheduler(self.optimizer, **scheduler_params)
                else:
                    self.scheduler = lr_scheduler(self.optimizer)

    def train(
        self,
        optimizer: Type[Optimizer],
        n_epochs: int = 10,
        cross_validation: bool = False,
        optimizers_param: Optional[Dict[str, Any]] = None,
        dataloaders_param: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        optuna_params: Optional[Tuple[BaseTrial, str]] = None,
        profiling: bool = False,
        parallel_tpu: bool = False,
        keep_training: bool = False,
        store_grad_layer_hist: bool = False,
        n_accumulated_grads: int = 0,
        writer_tag: str = "",
    ) -> Tuple[float, float]:
        """Function to run all the training cycles.

        Args:
            optimizer:
                the torch optimiser class, like `SGD`
            n_epochs :
                number of training epochs
            cross_validation:
                whether or not to perform five-fold cross-validation
            dataloaders_param:
                dictionary of the dataloaders
                parameters, e.g. `{'batch_size': 32}`. If ``None``, then
                the parameters of the training and validation
                dataloaders will be used.
            optimizers_param:
                dictionary of the optimizers
                parameters, e.g. `{"lr": 0.001}`
            lr_scheduler:
                a learning rate scheduler class
            scheduler_params:
                learning rate scheduler parameters
            optuna_params :
                the parameters `(trial, search_metric)`
                used in the gridsearch. Safely ignore for
                standard trainings
            profiling:
                whether or not you want to activate the
                profiler
            parallel_tpu:
                Use or not parallel TPU cores.
                Still experimental!
            keep_training:
                This flag allows to restart a training from
                the existing optimizer as well as the
                existing model
            store_grad_layer_hist:
                This flag allows to store the gradients
                and the layer values in tensorboard for
                each epoch
            n_accumulated_grads:
                this is the number of accumulated gradients.
                Only a positive number will be taken into account
            writer_tag:
                the tensorboard writer tag

        Returns:
            (float, float):
                the validation loss and accuracy
                if there is cross validation, the validation data loader
                is ignored. On the other hand, if there `cross_validation = False`
                then the test loss and accuracy is returned.
        """
        self.n_accumulated_grads = n_accumulated_grads
        self.store_grad_layer_hist = store_grad_layer_hist
        # to start the training from where we left
        if self.check_has_trained and keep_training:
            self._set_initial_model()

        # train initialisation
        dl_tr = self.dataloaders[0]
        if optimizers_param is None:
            optimizers_param = {"lr": 0.001}

        # dataloaders_param initialisation
        if dataloaders_param is None:
            if self.dataloaders[1] is not None:
                dataloaders_param_val = Trainer.copy_dataloader_params(
                    self.dataloaders[1]
                )
            else:
                dataloaders_param_val = Trainer.copy_dataloader_params(dl_tr)
            dataloaders_param_tr = Trainer.copy_dataloader_params(dl_tr)
        else:
            dataloaders_param_val = dataloaders_param.copy()
            dataloaders_param_tr = dataloaders_param.copy()

        # scheduler_params initialisation
        if scheduler_params is None:
            scheduler_params = {}

        # LR scheduler
        if not (self.check_has_trained and keep_training):
            self.scheduler = None

        # optuna gridsearch
        search_metric = None
        trial = None
        if optuna_params is not None:
            check_optuna = True
            trial, search_metric = optuna_params
        else:
            check_optuna = False

        # profiling
        self._init_profiler(
            profiling, cross_validation, n_epochs, self.k_fold_class.n_splits  # type: ignore
        )

        # remove sampler to avoid conflicts with indexing
        # we will re-introduce the sampler when creating the indexing list
        try:
            dataloaders_param_val.pop("sampler")
        except KeyError:
            pass

        try:
            dataloaders_param_tr.pop("sampler")
        except KeyError:
            pass

        # validation being the 20% in the case of 2
        # dataloders without crossvalidation
        if len(self.dataloaders) == 3:  # type: ignore
            try:
                val_idx = self.dataloaders[1].sampler.indices  # type: ignore
            except AttributeError:
                val_idx = list(range(len(self.dataloaders[1].dataset)))  # type: ignore
            # print(val_idx)
            dl_val = torch.utils.data.DataLoader(  # type: ignore
                self.dataloaders[1].dataset,
                # pin_memory=True,
                **dataloaders_param_val,
                sampler=SubsetRandomSampler(val_idx),
            )
            try:
                tr_idx = self.dataloaders[0].sampler.indices  # type: ignore
            except AttributeError:
                tr_idx = list(range(len(self.dataloaders[0].dataset)))  # type: ignore
            # print(tr_idx)
            dl_tr = torch.utils.data.DataLoader(  # type: ignore
                self.dataloaders[0].dataset,
                # pin_memory=True,
                **dataloaders_param_tr,
                sampler=SubsetRandomSampler(tr_idx),
            )
        else:
            try:
                data_idx = self.dataloaders[0].sampler.indices  # type: ignore
            except AttributeError:
                data_idx = list(range(len(self.dataloaders[0].dataset)))  # type: ignore
            # print(data_idx)
            tr_idx, val_idx = train_test_split(data_idx, test_size=0.2)
            dl_val = torch.utils.data.DataLoader(  # type: ignore
                self.dataloaders[0].dataset,
                # pin_memory=True,
                **dataloaders_param_val,
                sampler=SubsetRandomSampler(val_idx),
            )
            dl_tr = torch.utils.data.DataLoader(  # type: ignore
                self.dataloaders[0].dataset,
                # pin_memory=True,
                **dataloaders_param_tr,
                sampler=SubsetRandomSampler(tr_idx),
            )

        if cross_validation:
            mean_val_loss = []
            mean_val_acc = []
            try:
                data_idx = self.dataloaders[0].sampler.indices  # type: ignore
                labels_for_split = [
                    self.dataloaders[0].dataset[i][-1] for i in data_idx
                ]
            except AttributeError:
                data_idx = list(range(len(self.dataloaders[0].dataset)))  # type: ignore
                labels_for_split = [
                    self.dataloaders[0].dataset[i][-1] for i in data_idx
                ]

            for fold, (tr_idx, val_idx) in enumerate(
                self.k_fold_class.split(data_idx, labels_for_split)
            ):
                # prints for class balance
                # lab_tr_fold = [self.dataloaders[0].dataset[i][-1] for i in tr_idx]
                # lab_val_fold = [self.dataloaders[0].dataset[i][-1] for i in val_idx]
                # print("train labels:",[(i, lab_tr_fold.count(i)) for i in np.unique(np.array(lab_tr_fold))])
                # print("val labels:",[(i, lab_val_fold.count(i)) for i in np.unique(np.array(lab_val_fold))])
                # print("lenghts: ", len(lab_tr_fold), len(lab_val_fold))
                self._init_optimizer_and_scheduler(
                    keep_training,
                    cross_validation,
                    optimizer,
                    optimizers_param,
                    lr_scheduler,
                    scheduler_params,
                )

                # re-initialise data loaders
                if len(self.dataloaders) == 3:
                    warnings.warn(
                        "Validation set is ignored in automatic Cross Validation"
                    )
                dl_tr = torch.utils.data.DataLoader(  # type: ignore
                    self.dataloaders[0].dataset,
                    # pin_memory=True,
                    **dataloaders_param_tr,
                    sampler=SubsetRandomSampler(tr_idx),
                )
                dl_val = torch.utils.data.DataLoader(  # type: ignore
                    self.dataloaders[0].dataset,
                    # pin_memory=True,
                    **dataloaders_param_val,
                    sampler=SubsetRandomSampler(val_idx),
                )
                # print n-th fold
                print("\n\n********** Fold ", fold + 1, "**************")
                # the training and validation loop
                if parallel_tpu == False:
                    valloss, valacc = self._training_loops(
                        n_epochs,
                        dl_tr,
                        dl_val,
                        lr_scheduler,
                        self.scheduler,
                        check_optuna,
                        search_metric,
                        trial,
                        cross_validation,
                        writer_tag + "/fold = " + str(fold + 1),
                    )
                else:
                    valloss, valacc = self.parallel_tpu_training_loops(
                        n_epochs,
                        dl_tr,
                        dl_val,
                        optimizers_param,
                        lr_scheduler,
                        self.scheduler,
                        check_optuna,
                        search_metric,
                        trial,
                        cross_validation,
                    )

                mean_val_loss.append(valloss)
                mean_val_acc.append(valacc)
            # mean of the validation and loss accuracies over folds
            if self.best_not_last:
                valloss = min(
                    np.array(
                        _inner_refactor_scalars(
                            self.val_loss_list_hparam, True, self.k_fold_class.n_splits  # type: ignore
                        )
                    )[:, 0]
                )
                valacc = max(
                    np.array(
                        _inner_refactor_scalars(
                            self.val_acc_list_hparam, True, self.k_fold_class.n_splits  # type: ignore
                        )
                    )[:, 0]
                )
            else:
                valloss = torch.mean(torch.tensor(mean_val_loss)).item()
                valacc = torch.mean(torch.tensor(mean_val_acc)).item()

        else:
            self._init_optimizer_and_scheduler(
                keep_training,
                cross_validation,
                optimizer,
                optimizers_param,
                lr_scheduler,
                scheduler_params,
            )

            if not parallel_tpu:
                valloss, valacc = self._training_loops(
                    n_epochs,
                    dl_tr,
                    dl_val,
                    lr_scheduler,
                    self.scheduler,
                    check_optuna,
                    search_metric,
                    trial,
                    False,
                    writer_tag,
                )
            else:
                valloss, valacc = self.parallel_tpu_training_loops(
                    n_epochs,
                    dl_tr,
                    dl_val,
                    optimizers_param,
                    lr_scheduler,
                    self.scheduler,
                    check_optuna,
                    search_metric,
                    trial,
                    False,
                )

        try:
            self.writer.flush()  # type: ignore
        except AttributeError:
            pass
        # check for training
        self.check_has_trained = True

        # put the mean of the cross_val
        return valloss, valacc

    def _training_loops(
        self,
        n_epochs: int,
        dl_tr: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        dl_val: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler: Optional[_LRScheduler] = None,
        check_optuna: bool = False,
        search_metric: Optional[str] = None,
        trial: Optional[BaseTrial] = None,
        cross_validation: bool = False,
        writer_tag: str = "",
    ) -> Tuple[float, float]:
        """private method to run the trainign loops

        Args:
            n_epochs:
                number of training epochs
            dl_tr:
                training dataloader
            dl_val:
                validation dataloader
                parameters, e.g. `{'batch_size': 32}`
            lr_scheduler:
                a learning rate scheduler class
            scheduler:
                the actual scheduler instance
            check_optuna:
                boolean to store the optuna results of
                the trial
            search_metric:
                either ``'loss'`` or ``'accuracy'``, this
                corresponds to the gridsearch criterion
            trial:
                the optuna trial
            cross_validation (bool, default False)
                the boolean flag for cross validation
            writer_tag:
                the tensorboard writer tag

        Returns:
            (float, float):
                the validation loss and validation accuracy
        """

        valloss, valacc = np.inf, 0.0
        for t in range(n_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.val_epoch = t
            self.train_epoch = t
            self._train_loop(dl_tr, writer_tag)
            me = ModelExtractor(self.model, self.loss_fn)
            if self.store_grad_layer_hist:
                try:
                    lista = me.get_layers_param()
                    for k, item in lista.items():
                        try:
                            self.writer.add_histogram(  # type: ignore
                                writer_tag + "/weights&biases/param/train/" + k, item, t
                            )
                            self.writer.add_histogram(  # type: ignore
                                writer_tag + "/weights&biases/param/train/log/" + k,
                                torch.log(torch.abs(item) + 1e-8),
                                t,
                            )
                        except ValueError:
                            warnings.warn(
                                f"The histogram is empty, most likely because your loss"
                                f" is exploding. Try use gradient clipping."
                            )
                    lista_grad = me.get_layers_grads()
                    for k, item in zip(lista.keys(), lista_grad):
                        try:
                            self.writer.add_histogram(  # type: ignore
                                writer_tag + "/weights&biases/grads/train/" + k, item, t
                            )
                            self.writer.add_histogram(  # type: ignore
                                writer_tag + "/weights&biases/param/train/log/" + k,
                                torch.log(torch.abs(item) + 1e-8),
                                t,
                            )
                        except ValueError:
                            warnings.warn(
                                f"The histogram is empty, most likely because your loss"
                                f" is exploding. Try use gradient clipping."
                            )
                    self.writer.flush()  # type: ignore

                except AttributeError:
                    pass
            self._run_pipe_hook(t + 1, self.optimizer, me, self.writer)
            valloss, valacc = self._val_loop(dl_val, writer_tag)
            # absolute best of loss and accuracy
            self.best_val_acc = max(self.best_val_acc, valacc)
            self.best_val_loss = min(self.best_val_loss, valloss)
            # print(self.optimizer.param_groups[0]["lr"])
            if not lr_scheduler is None:
                scheduler.step()  # type: ignore
            # pruning trials
            if check_optuna and not cross_validation:
                if search_metric == "loss":
                    trial.report(valloss, t)  # type: ignore
                else:
                    trial.report(valacc, t)  # type: ignore
                # Handle pruning based on the intermediate value.
                if trial.should_prune():  # type: ignore
                    raise optuna.exceptions.TrialPruned()
        return valloss, valacc

    def parallel_tpu_training_loops(
        self,
        n_epochs: int,
        dl_tr_old: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        dl_val_old: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]],
        optimizers_param: Dict[str, Any],
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler: Optional[_LRScheduler] = None,
        check_optuna: bool = False,
        search_metric: Optional[str] = None,
        trial: Optional[BaseTrial] = None,
        cross_validation: bool = False,
    ) -> Tuple[float, float]:
        """Experimental function to run all the training cycles
        on colab TPUs in parallel.
        Note: ``cross_validation`` parameter as well as
        ``profiling`` are ignored.

        Args:
            n_epochs:
                number of training epochs
            dl_tr_old:
                training dataloader
            dl_val_old:
                validation dataloader
                parameters, e.g. `{'batch_size': 32}`
            optimizers_param:
                dictionary of parameters for the optimizers
            lr_scheduler :
                a learning rate scheduler class
            scheduler:
                the actual scheduler instance
            check_optuna :
                boolean to store the optuna results of
                the trial
            search_metric:
                either ``'loss'`` or ``'accuracy'``, this
                corresponds to the gridsearch criterion
            trial:
                the optuna trial
            cross_validation :
                the boolean flag for cross validation

        Returns:
            (float, float):
                the validation loss and validation accuracy
        """
        self.val_loss = 0
        self.val_acc = 0
        warnings.warn(
            "The tensorboard writer is ignored "
            + "for multi TPU processing. Also SAM optimisation"
            + " does not work for multi TPU training."
        )

        def map_fun_custom(index, flags):
            """map function for multi-processing"""
            device = xm.xla_device()

            print("uploading model to TPU")
            model2 = self.model.to(device)

            # initialize optimizer
            optimizer_class = type(self.optimizer)
            optimizer = optimizer_class(model2.parameters(), **optimizers_param)

            # define training and validation
            # distributed samplers and update
            # the dataloaders
            train_sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore
                dl_tr_old.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True,
            )

            dl_tr = torch.utils.data.DataLoader(  # type: ignore
                dl_tr_old.dataset,
                num_workers=dl_tr_old.num_workers,
                batch_size=dl_tr_old.batch_size,
                sampler=train_sampler,
                drop_last=dl_tr_old.drop_last,
            )

            val_sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore
                dl_val_old.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False,
            )

            dl_val = torch.utils.data.DataLoader(  # type: ignore
                dl_val_old.dataset,
                num_workers=dl_val_old.num_workers,
                batch_size=dl_val_old.batch_size,
                sampler=val_sampler,
                drop_last=dl_val_old.drop_last,
            )

            # train loop
            for t in range(n_epochs):
                model2.train()
                para_train_loader = pl.ParallelLoader(
                    dl_tr, [device]
                ).per_device_loader(device)
                print(f"Epoch {t + 1}\n-------------------------------")
                self.val_epoch = t
                self.train_epoch = t

                # train batch loop
                loss = 0.0
                correct = 0.0
                tik = time.time()
                # for batch, (X, y) in enumerate(self.dataloaders[0]):
                for batch, (X, y) in enumerate(para_train_loader):
                    # Compute prediction and loss
                    pred = model2(X)
                    try:
                        correct += (
                            (pred.argmax(1) == y).to(torch.float).sum().item()
                        )  # noqa
                    except RuntimeError:
                        correct += (
                            (pred.argmax(2) == y).to(torch.float).sum().item()
                        )  # noqa
                    loss = self.loss_fn(pred, y)
                    # Save to tensorboard
                    # self.writer.add_scalar("Parallel" + "/Loss/train",
                    #                       loss.cpu(),
                    #                       self.train_epoch*dl_tr.batch_size + batch)
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()

                    xm.optimizer_step(self.optimizer)

                # train accuracy:
                correct /= len(dl_tr) * dl_tr.batch_size
                print("Train accuracy at epoch ", t, " : ", correct)
                print(f"\nTime taken for this epoch: {round(time.time() - tik):.2f}s")

                # evaluation
                model2.eval()

                loss, correct = 0.0, 0.0
                class_label: List[Tensor] = []
                class_probs: List[List[Tensor]] = []

                pred = 0.0
                para_valid_loader = pl.ParallelLoader(
                    dl_val, [device]
                ).per_device_loader(device)
                with torch.no_grad():
                    # per batch!!
                    for X, y in para_valid_loader:
                        pred = model2(X)
                        class_probs_batch = [f.softmax(el, dim=0) for el in pred]
                        class_probs.append(class_probs_batch)
                        loss += self.loss_fn(pred, y).item()
                        try:
                            correct += (
                                (pred.argmax(1) == y).to(torch.float).sum().item()
                            )
                        except RuntimeError:
                            correct += (
                                (pred.argmax(2) == y).to(torch.float).sum().item()
                            )
                        class_label.append(y)
                    # add data to tensorboard
                    # self._add_pr_curve_tb(pred, class_label, class_probs, "validation")

                # self.writer.add_scalar("Parallel " + "/Accuracy/validation", correct, self.val_epoch)
                print(
                    f"Validation results: \n Accuracy: {(100 * correct):.2f}%, \
                        Avg loss: {loss:>8f} \n"
                )

                # self.writer.flush()

                if lr_scheduler is not None:
                    assert scheduler is not None, "scheduler is None"
                    scheduler.step()
                if self.prof is not None:
                    self.prof.step()

                if check_optuna and not cross_validation:
                    if search_metric == "loss":
                        assert trial is not None, "trial is None"
                        trial.report(loss, t)
                    else:
                        assert trial is not None, "trial is None"
                        trial.report(correct, t)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            self.val_loss += loss  # type: ignore
            self.val_acc += correct * 100  # type: ignore

        flags: Dict[Any, Any] = {}
        self.val_acc /= len(dl_val_old) * dl_val_old.batch_size  # type: ignore
        self.val_loss /= len(dl_val_old)  # type: ignore
        xmp.spawn(map_fun_custom, args=(flags,), nprocs=8, start_method="fork")
        return self.val_loss, self.val_acc

    def evaluate_classification(
        self,
        num_class: Optional[int] = None,
        dl: Optional[DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]]] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """Method to evaluate the performance of the model.

        Args:
            num_class:
                number of classes
            dl :
                the Dataloader to evaluate. If ``None``,
                we use the training dataloader in ``self``

        Returns:
            (float, float, 2darray):
                the accuracy, loss and confusion matrix.
        """
        if dl is None:
            dl = self.dataloaders[0]
        class_probs: List[List[Tensor]] = []
        class_label: List[Tensor] = []
        batch_metric_list = []
        loss = 0.0
        correct = 0.0
        confusion_matrix = np.zeros((num_class, num_class))  # type: ignore
        self.model.eval()
        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(dl)):
                pred, X, y = self._send_to_device(X, y)
                class_probs_batch = [f.softmax(el, dim=0) for el in pred]
                class_probs.append(class_probs_batch)
                loss += self.loss_fn(pred, y).item()
                batch_metric = self.training_metric(pred, y)
                batch_metric_list.append(batch_metric)
                class_label.append(y)
                for t, p in zip(y.view(-1), pred.argmax(1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        epoch_metric = sum(batch_metric_list) / len(batch_metric_list)
        epoch_loss = loss / len(batch_metric_list)
        return epoch_metric, epoch_loss, confusion_matrix

    def register_pipe_hook(
        self,
        callable: Callable[
            [int, Optimizer, ModelExtractor, Optional[SummaryWriter]], Any
        ],
    ) -> None:
        """This method registers a function that
        will be called after each trainign step.

        The arguments of the callable function are, in this order:
         - current epoch number
         - current optimizer instance
         - the ModelExtractor instance at that epoch
         - the tensorboard writer

        Args:
            callable (Callable):
                the function to register"""
        self.registered_hook = callable

    def _run_pipe_hook(
        self,
        epoch: int,
        optim: Optimizer,
        me: ModelExtractor,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        """private method that runs the hooked
        function at every epoch, after the single training loop"""
        if self.registered_hook is not None:
            self.registered_hook(epoch, optim, me, writer)

    @staticmethod
    def copy_dataloader_params(
        original_dataloader: DataLoader[Tuple[Union[Tensor, List[Tensor]], Tensor]]
    ) -> Dict[str, Any]:
        """returns the dict of init parameters"""
        return {
            "batch_size": original_dataloader.batch_size,
            # "batch_sampler": original_dataloader.batch_sampler,
            "num_workers": original_dataloader.num_workers,
            "collate_fn": original_dataloader.collate_fn,
            "pin_memory": original_dataloader.pin_memory,
            "drop_last": original_dataloader.drop_last,
            "timeout": original_dataloader.timeout,
            "worker_init_fn": original_dataloader.worker_init_fn,
            "multiprocessing_context": original_dataloader.multiprocessing_context,
            "generator": original_dataloader.generator,
            "prefetch_factor": original_dataloader.prefetch_factor,
            "persistent_workers": original_dataloader.persistent_workers,
        }

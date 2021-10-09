import torch.nn.functional as F
import torch
import numpy as np
import copy
import time
import warnings
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.data import PreprocessText
import optuna

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU!")
else:
    DEVICE = torch.device("cpu")
    
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    DEVICE = xm.xla_device()
    print("Using TPU!")
except:
    print("No TPUs...")

class Pipeline:
    """This is the generic class that allows
    the user to benchhmark models over architectures
    datasets, regularisations, metrics... in one line
    of code.

    Args:
        model (nn.Module):
            standard torch model
        dataloaders (list of utils.DataLoader):
            list of standard torch DtaLoaders, e.g.
            `[dl_tr, dl_val, dl_ts]`
        loss_fn (Callables):
            loss function to average over batches
        wirter (tensorboard SummaryWriter):
            tensorboard writer

    """

    # def __init__(self, model, dataloaders, loss_fn, writer,
    # hyperparams_search = False, search_metric = "accuracy", n_trials = 10):
    def __init__(self, model, dataloaders, loss_fn, writer):
        self.model = model.to(DEVICE)
        self.initial_model = copy.deepcopy(self.model).to(DEVICE)
        assert len(dataloaders) > 0 and len(dataloaders) < 4, "Length of dataloaders must be 1, 2, or 3"
        self.dataloaders = dataloaders  # train and test
        self.train_epoch = 0
        self.val_epoch = 0
        self.loss_fn = loss_fn
        # integrate tensorboard
        self.writer = writer
        
    def reset_model(self):
        """method to reset the initial model weights. This
        function is essential for the cross-validation
        procedure.
        """

        self.model = copy.deepcopy(self.initial_model).to(DEVICE)

            

    def reset_epoch(self):
        """method to reset global training and validation
        epoch count.
        """

        self.train_epoch = 0
        self.val_epoch = 0

    def _train_loop(self, dl_tr, writer_tag=""):
        """private method to run a single training
        loop
        """
        self.model.train()
        size = len(dl_tr.dataset)
        steps = len(dl_tr)
        loss = -100    # arbitrary starting value to avoid nan loss
        correct = 0
        tik = time.time()
        # for batch, (X, y) in enumerate(self.dataloaders[0]):
        for batch, (X, y) in enumerate(dl_tr):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            # Compute prediction and loss
            pred = self.model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss = self.loss_fn(pred, y)
            # Save to tensorboard
            self.writer.add_scalar(writer_tag + "/Loss/train",
                                   loss,
                                   self.train_epoch*dl_tr.batch_size + batch)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            if DEVICE.type == "xla":
                xm.optimizer_step(self.optimizer, barrier=True)  # Note: Cloud TPU-specific code!
            else:
                self.optimizer.step()
            if batch % 1 == 0:
                t_loss = loss.item()
                print("Training loss: ", t_loss, " [",batch+1,"/",
                      steps,"]                     ", end='\r')
        self.writer.flush()
        # accuracy:
        correct /= size
        print("\nTime taken for this epoch: {}s".format(round(time.time()-tik), 2))
        return loss, correct

    def _val_loop(self, dl_val, writer_tag="validation"):
        """private method to run a single validation
        loop
        """

        # size = len(self.dataloaders[1].dataset)
        size = len(dl_val.dataset)
        val_loss, correct = 0, 0
        class_label = []
        class_probs = []
        self.model.eval()

        pred, val_loss, correct = self._inner_loop(dl_val,
                                                   class_probs,
                                                   class_label,
                                                   val_loss,
                                                   correct)
        # add data to tensorboard
        self._add_pr_curve_tb(pred, class_label, class_probs, writer_tag)
        self.writer.flush()

        # accuracy
        val_loss /= size
        correct /= size

        self.writer.add_scalar(writer_tag + "/Accuracy/validation", correct, self.val_epoch)
        print(f"Validation results: \n Accuracy: {(100*correct):>0.1f}%, \
                Avg loss: {val_loss:>8f} \n")

        self.writer.flush()

        return val_loss, 100*correct

    def _add_pr_curve_tb(self, pred, class_label, class_probs, writer_tag=""):
        """private function to add the PR curve
        to tensorboard"""
        probs = torch.cat([torch.stack(batch) for batch in
                          class_probs]).cpu()
        labels = torch.cat(class_label).cpu()
        for class_index in range(len(pred[0])):
            tensorboard_truth = labels == class_index
            tensorboard_probs = probs[:, class_index]
            self.writer.add_pr_curve(writer_tag+str(class_index),
                                     tensorboard_truth,
                                     tensorboard_probs,
                                     global_step=0)

    def _inner_loop(self, dl, class_probs, class_label, loss, correct):
        """private function used inside the test
        and validation loops"""
        pred = 0
        for X, y in dl:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = self.model(X)
            class_probs_batch = [F.softmax(el, dim=0)
                                 for el in pred]
            class_probs.append(class_probs_batch)
            loss += self.loss_fn(pred, y).item()
            correct += (pred.argmax(1) ==
                        y).type(torch.float).sum().item()
            class_label.append(y)
        return pred, correct, loss

    def _test_loop(self, dl_test, writer_tag="test"):
        """private method to run a single test
        loop
        """

        # size = len(self.dataloaders[1].dataset)
        size = len(dl_test.dataset)
        test_loss, correct = 0, 0
        class_label = []
        class_probs = []
        self.model.eval()

        # for X, y in self.dataloaders[1]:
        pred, test_loss, correct = self._inner_loop(dl_test,
                                                    class_probs,
                                                    class_label,
                                                    test_loss,
                                                    correct)
        # add data to tensorboard
        self._add_pr_curve_tb(pred, class_label, class_probs, writer_tag)

        self.writer.flush()

        # accuracy
        correct /= size
        test_loss /= size
        print(f"Test results: \n Accuracy: {(100*correct):>0.1f}%, \
                Avg loss: {test_loss:>8f} \n")

        return test_loss, 100*correct

    def train(self, optimizer, n_epochs=10, cross_validation=False,
              optimizers_param=None,
              dataloaders_param=None,
              lr_scheduler=None,
              scheduler_params=None,
              optuna_params=None,
              profiling=False,
              k_folds=5,
              parallel_tpu=False):
        """Function to run all the training cycles.

        Args:
            optimizer (torch.optim):
                the torch optimiser class, like `SGD`
            n_epochs (int):
                number of training epochs
            cross_validation (bool):
                whether or not to perform five-fold cross-validation
            dataloaders_param (dict):
                dictionary of the dataloaders
                parameters, e.g. `{'batch_size': 32}`
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
            optuna_params (tuple, default=None):
                the parameters `(trial, search_metric)`
                used in the gridsearch. Saefly ignore for
                standard trainings
            profiling (bool, default=False):
                whether or not you want to activate the
                profiler
            k_folds (int, default=5):
                number of folds in cross validation
            parallel_tpu (bool):
                Use or not parallel TPU cores.
                Still experimental!
            
        Returns:
            (float, float):
                the validation loss and accuracy
                if there is cross validation, the validation data loader
                is ignored. On the other hand, if there `cross_validation = False`
                then the test loss and accuracy is returned.
        """
        # train initialisation
        dl_tr = self.dataloaders[0]
        if optimizers_param is None:
            optimizers_param = {"lr":0.001}
        if dataloaders_param is None:
            dataloaders_param = {"batch_size":dl_tr.batch_size}
        if scheduler_params is None:
            scheduler_params = {}
        
        # LR scheduler
        scheduler = None
        
        # optuna gridsearch
        search_metric = None
        trial = None
        if not optuna_params is None:
            check_optuna = True
            trial, search_metric = optuna_params
        else:
            check_optuna = False
        
        # profiling
        prof = None
        if not cross_validation:
            active = n_epochs-2
        else:
            active = k_folds*(n_epochs-2)
        if profiling:
            try:
                prof = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=active),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs',
                                                                                worker_name='worker'),
                        record_shapes=True,
                        #profile_memory=True,
                        with_stack=True
                )
            except AssertionError:
                pass
        
        # validation being the test set for 2
        # dataloders without crossvalidation
        dl_val = self.dataloaders[1]

        if cross_validation:
            mean_val_loss = []
            mean_val_acc = []
            valloss, valacc = -1, 0
            data_idx = list(range(len(self.dataloaders[0])*self.dataloaders[0].batch_size))
            fold = KFold(k_folds, shuffle=False)
            for fold, (tr_idx, val_idx) in enumerate(fold.split(data_idx)):
                # reset the model weights
                self.reset_model()
                self.optimizer = optimizer(self.model.parameters(), **optimizers_param)
                if not lr_scheduler is None:
                    scheduler = lr_scheduler(self.optimizer, **scheduler_params)
                # re-initialise data loaders
                if len(self.dataloaders) == 3:
                    warnings.warn("Validation set is ignored in automatic Cross Validation")
                dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                    shuffle=False,
                                                    #pin_memory=True,
                                                    **dataloaders_param,
                                                    sampler=SubsetRandomSampler(tr_idx))
                dl_val = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                     shuffle=False,
                                                     #pin_memory=True,
                                                     **dataloaders_param,
                                                     sampler=SubsetRandomSampler(val_idx))
                # print n-th fold
                print("\n\n********** Fold ", fold+1, "**************")
                # the training and validation loop
                if parallel_tpu == False:
                    valloss, valacc = self._training_loops(n_epochs, dl_tr,
                                                       dl_val, lr_scheduler, scheduler,
                                                       prof, check_optuna, search_metric,
                                                       trial)
                else:
                    valloss, valacc = self.parallel_tpu_training_loops(n_epochs, dl_tr,
                                                       dl_val, lr_scheduler, scheduler,
                                                       prof, check_optuna, search_metric,
                                                       trial)
                
                mean_val_loss.append(valloss)
                mean_val_acc.append(valacc)
                # mean of the validation and loss accuracies over folds
            valloss = np.mean(mean_val_loss)
            valacc = np.mean(mean_val_acc)


        else:
            if not dataloaders_param == {}:
                dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                    shuffle=False,
                                                    #pin_memory=True,
                                                    sampler=range(len(dl_tr)*dl_tr.batch_size),
                                                    **dataloaders_param)
            self.reset_model()
            self.optimizer = optimizer(self.model.parameters(), **optimizers_param)
            if not lr_scheduler is None:
                scheduler = lr_scheduler(self.optimizer, **scheduler_params)

            if parallel_tpu == False:
                valloss, valacc = self._training_loops(n_epochs, dl_tr,
                                                   dl_val, lr_scheduler, scheduler,
                                                   prof, check_optuna, search_metric,
                                                   trial)
            else:
                valloss, valacc = self.parallel_tpu_training_loops(n_epochs, dl_tr,
                                                   dl_val, lr_scheduler, scheduler,
                                                   prof, check_optuna, search_metric,
                                                   trial)

        self.writer.flush()
        # put the mean of the cross_val
        
        return valloss, valacc
        
    def _training_loops(self, n_epochs, dl_tr,
                        dl_val, lr_scheduler, scheduler,
                        prof, check_optuna, search_metric,
                        trial):
        """private method to run the trainign loops
        
        Args:
            n_epochs (int):
                number of training epochs
            dl_tr (torch.DataLoader):
                training dataloader
            dl_val (torch.DataLoader):
                validation dataloader
                parameters, e.g. `{'batch_size': 32}`
            optimizers_param (dict):
                dictionary of the optimizers
                parameters, e.g. `{"lr": 0.001}`
            models_param (dict):
                dictionary of the model
                parameters
            lr_scheduler (torch.optim):
                a learning rate scheduler
            scheduler (torch.optim):
                the actual scheduler
            prof (bool, default=False):
                whether or not you want to activate the
                profiler
            check_optuna (bool):
                boolean to store the optuna results of
                the trial
            search_metric (string):
                either ``'loss'`` or ``'accuracy'``, this
                corresponds to the gridsearch criterion
            trial (optuna.trial):
                the optuna trial

        Returns:
            (float, float):
                the validation loss and validation accuracy
        """

        valloss, valacc = 100, 0
        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.val_epoch = t
            self.train_epoch = t
            self._train_loop(dl_tr, "train")
            valloss, valacc = self._val_loop(dl_val, "validation")
            
            #print(self.optimizer.param_groups[0]["lr"])
            if not lr_scheduler is None:
                scheduler.step()
            if not prof is None:
                prof.step()

            if check_optuna:
                if search_metric == "loss":
                    trial.report(valloss, t)
                else:
                    trial.report(valacc, t)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return valloss, valacc


    def parallel_tpu_training_loops(self, n_epochs, dl_tr,
                                    dl_val, lr_scheduler, scheduler,
                                    prof, check_optuna, search_metric,
                                    trial):
        """Experimental function to run all the training cycles
        on colab TPUs in parallel.
        Note: ``cross_validation`` parameter as well as
        ``profiling`` are ignored.

        Args:
            n_epochs (int):
                number of training epochs
            dl_tr (torch.DataLoader):
                training dataloader
            dl_val (torch.DataLoader):
                validation dataloader
                parameters, e.g. `{'batch_size': 32}`
            optimizers_param (dict):
                dictionary of the optimizers
                parameters, e.g. `{"lr": 0.001}`
            models_param (dict):
                dictionary of the model
                parameters
            lr_scheduler (torch.optim):
                a learning rate scheduler
            scheduler (torch.optim):
                the actual scheduler
            prof (bool, default=False):
                whether or not you want to activate the
                profiler
            check_optuna (bool):
                boolean to store the optuna results of
                the trial
            search_metric (string):
                either ``'loss'`` or ``'accuracy'``, this
                corresponds to the gridsearch criterion
            trial (optuna.trial):
                the optuna trial

        Returns:
            (float, float):
                the validation loss and validation accuracy
        """
        valloss, valacc = 100, 0
        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.val_epoch = t
            self.train_epoch = t
            self._train_loop(dl_tr, "train")
            valloss, valacc = self._val_loop(dl_val, "validation")
            
            #print(self.optimizer.param_groups[0]["lr"])
            if not lr_scheduler is None:
                scheduler.step()
            if not prof is None:
                prof.step()

            if check_optuna:
                if search_metric == "loss":
                    trial.report(valloss, t)
                else:
                    trial.report(valacc, t)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        def map_fun_custom(index, flags):
            """map function for multi-processing"""
            device = xm.xla_device()

            print("uploading net to device")
            self.model.to(device)

            #train loop
            for t in range(flags['num_epochs']):
                self.model.train()
                para_train_loader = pl.ParallelLoader(dl_tr,
                                                      [device]).per_device_loader(device)
                print(f"Epoch {t+1}\n-------------------------------")
                self.val_epoch = t
                self.train_epoch = t
                # train batch loop
                
                size = len(dl_tr.dataset)
                steps = len(dl_tr)
                loss = -100    # arbitrary starting value to avoid nan loss
                correct = 0
                tik = time.time()
                # for batch, (X, y) in enumerate(self.dataloaders[0]):
                for batch, (X, y) in enumerate(para_train_loader):
                    # Compute prediction and loss
                    pred = self.model(X)
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    loss = self.loss_fn(pred, y)
                    # Save to tensorboard
                    self.writer.add_scalar("train" + "/Loss/train",
                                           loss,
                                           self.train_epoch*dl_tr.batch_size + batch)
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()

                    xm.optimizer_step(self.optimizer)

                self.writer.flush()
                # train accuracy:
                correct /= size
                print("\nTime taken for this epoch: {}s".format(round(time.time()-tik), 2))

                # evaluation
                self.model.eval()

                # size = len(self.dataloaders[1].dataset)
                size = len(dl_val.dataset)
                val_loss, correct = 0, 0
                class_label = []
                class_probs = []

                pred = 0
                para_valid_loader = pl.ParallelLoader(dl_val,
                                                      [device]).per_device_loader(device)
                for X, y in para_valid_loader:
                    X = X.to(device)
                    y = y.to(device)
                    pred = self.model(X)
                    class_probs_batch = [F.softmax(el, dim=0)
                                         for el in pred]
                    class_probs.append(class_probs_batch)
                    loss += self.loss_fn(pred, y).item()
                    correct += (pred.argmax(1) ==
                                y).type(torch.float).sum().item()
                    class_label.append(y)
                # add data to tensorboard
                self._add_pr_curve_tb(pred, class_label, class_probs, "validation")
                self.writer.flush()

                # validation accuracy
                loss /= size
                correct /= size

                self.writer.add_scalar(writer_tag + "/Accuracy/validation", correct, self.val_epoch)
                print(f"Validation results: \n Accuracy: {(100*correct):>0.1f}%, \
                        Avg loss: {val_loss:>8f} \n")

                self.writer.flush()
              
                if not lr_scheduler is None:
                    scheduler.step()
                if not prof is None:
                    prof.step()

                if check_optuna:
                    if search_metric == "loss":
                        trial.report(valloss, t)
                    else:
                        trial.report(valacc, t)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        flags = {}

        xmp.spawn(map_fun_custom, args=(flags,), nprocs=8, start_method='fork')
        return loss, correct

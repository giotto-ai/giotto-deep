import torch.nn.functional as F
import torch
import numpy as np
import copy
import time
from tqdm import tqdm
import warnings
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
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
            list of standard torch DataLoaders, e.g.
            `[dl_tr, dl_val, dl_ts]`
        loss_fn (Callables):
            loss function to average over batches
        wirter (tensorboard SummaryWriter):
            tensorboard writer

    """

    # def __init__(self, model, dataloaders, loss_fn, writer,
    # hyperparams_search = False, search_metric = "accuracy", n_trials = 10):
    def __init__(self, model, dataloaders, loss_fn, writer):
        self.model = model
        self.initial_model = copy.deepcopy(self.model)
        assert len(dataloaders) > 0 and len(dataloaders) < 4, "Length of dataloaders must be 1, 2, or 3"
        self.dataloaders = dataloaders  # train and test
        self.train_epoch = 0
        self.val_epoch = 0
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.loss_fn = loss_fn
        # integrate tensorboard
        self.writer = writer
        if self.writer is None:
            warnings.warn("No writer detected")
        self.DEVICE = DEVICE
        
    def reset_model(self):
        """method to reset the initial model weights. This
        function is essential for the cross-validation
        procedure.
        """

        self.model = copy.deepcopy(self.initial_model)

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
        try:
            self.DEVICE = xm.xla_device()
        except NameError:
            print("No TPUs")
        self.model = self.model.to(self.DEVICE)
        self.model.train()
        size = len(dl_tr.dataset)
        steps = len(dl_tr)
        loss = 0
        correct = 0
        t_loss = 0
        tik = time.time()
        # for batch, (X, y) in enumerate(self.dataloaders[0]):
        for batch, (X, y) in enumerate(dl_tr):
            X = X.to(self.DEVICE)
            y = y.to(self.DEVICE)
            # Compute prediction and loss
            pred = self.model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss = self.loss_fn(pred, y)
            # Save to tensorboard
            try:
                self.writer.add_scalar(writer_tag + "/loss/train",
                                       loss.item(),
                                       self.train_epoch*len(dl_tr) + batch)
            except AttributeError:
                pass
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            if self.DEVICE.type == "xla":
                xm.optimizer_step(self.optimizer, barrier=True)  # Note: Cloud TPU-specific code!
            else:
                self.optimizer.step()
            if batch % 1 == 0:
                t_loss += loss.item()
                print("Batch training loss: ", t_loss/(batch+1), " \tBatch training accuracy: ",
                      correct/((batch+1)*dl_tr.batch_size)*100,
                      " \t[",batch+1,"/", steps,"]                     ", end='\r')
        try:
            self.writer.flush()
        except AttributeError:
            pass
        # accuracy:
        correct /= len(dl_tr)*dl_tr.batch_size
        t_loss /= len(dl_tr)
        print("\nTime taken for this epoch: {}s".format(round(time.time()-tik), 2))
        return t_loss, correct*100

    def _val_loop(self, dl_val, writer_tag=""):
        """private method to run a single validation
        loop
        """
        try:
            self.DEVICE = xm.xla_device()
        except NameError:
            print("No TPUs")
        self.model = self.model.to(self.DEVICE)
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
        try:
            # add data to tensorboard
            self._add_pr_curve_tb(pred, class_label, class_probs, writer_tag + "/validation")
            try:
                self.writer.flush()
            except AttributeError:
                pass
        except NotImplementedError:
            warnings.warn("The PR curve is not being filled because too few data exist")
        # accuracy
        correct /= len(dl_val)*dl_val.batch_size
        val_loss /= len(dl_val)
        try:
            self.writer.add_scalar(writer_tag + "/accuracy/validation", correct, self.val_epoch)
        except AttributeError:
            pass
        print(f"Validation results: \n Accuracy: {(100*correct):>8f}%, \
                Avg loss: {val_loss:>8f} \n")
        try:
            self.writer.flush()
        except AttributeError:
            pass
        # store the best results
        self.best_val_acc = max(self.best_val_acc,100*correct)
        self.best_val_loss = min(self.best_val_loss,100*correct)
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
            try:
                self.writer.add_pr_curve(writer_tag+"/class = "+str(class_index),
                                         tensorboard_truth,
                                         tensorboard_probs,
                                         global_step=0)
            except AttributeError:
                pass
    def _inner_loop(self, dl, class_probs, class_label,
                    loss, correct):
        """private function used inside the test
        and validation loops"""
        pred = 0
        with torch.no_grad():
            for X, y in dl:
                X = X.to(self.DEVICE)
                y = y.to(self.DEVICE)
                pred = self.model(X)
                class_probs_batch = [F.softmax(el, dim=0)
                                     for el in pred]
                class_probs.append(class_probs_batch)
                loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) ==
                            y).type(torch.float).sum().item()
                class_label.append(y)
        return pred, loss, correct

    def _test_loop(self, dl_test, writer_tag=""):
        """private method to run a single test
        loop
        """
        try:
            self.DEVICE = xm.xla_device()
        except NameError:
            print("No TPUs")
        self.model = self.model.to(self.DEVICE)
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
        self._add_pr_curve_tb(pred, class_label, class_probs, writer_tag + "/test")
        try:
            self.writer.flush()
        except AttributeError:
            pass

        # accuracy
        correct /= len(dl_test)*dl_test.batch_size
        test_loss /= len(dl_test)
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
              parallel_tpu=False,
              writer_tag=""):
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
            writer_tag (str):
                the tensorboard writer tag

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
        
        # remove sampler to avoid conflicts with validation
        dataloaders_param_val = dataloaders_param.copy()
        try:
            dataloaders_param_val.pop("sampler")
        except KeyError:
            pass
        
        # validation being the 20% in the case of 2
        # dataloders without crossvalidation
        if len(self.dataloaders) == 3:
            val_idx = list(range((len(self.dataloaders[1])-1)*self.dataloaders[1].batch_size))
            dl_val = torch.utils.data.DataLoader(self.dataloaders[1].dataset,
                                                 shuffle=False,
                                                 #pin_memory=True,
                                                 **dataloaders_param_val,
                                                 sampler=SubsetRandomSampler(val_idx))
            tr_idx = list(range((len(self.dataloaders[0])-1)*self.dataloaders[0].batch_size))
            dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                shuffle=False,
                                                #pin_memory=True,
                                                **dataloaders_param_val,
                                                sampler=SubsetRandomSampler(tr_idx))
        else:
            
            data_idx = list(range((len(self.dataloaders[0])-1)*self.dataloaders[0].batch_size))
            tr_idx, val_idx = train_test_split(data_idx, test_size=0.2)
            dl_val = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                 shuffle=False,
                                                 #pin_memory=True,
                                                 **dataloaders_param_val,
                                                 sampler=SubsetRandomSampler(val_idx))
            dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                shuffle=False,
                                                #pin_memory=True,
                                                **dataloaders_param_val,
                                                sampler=SubsetRandomSampler(tr_idx))

        if cross_validation:
            mean_val_loss = []
            mean_val_acc = []
            valloss, valacc = 0, 0
            data_idx = list(range((len(self.dataloaders[0])-1)*self.dataloaders[0].batch_size))
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
                                                    **dataloaders_param_val,
                                                    sampler=SubsetRandomSampler(tr_idx))
                dl_val = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                     shuffle=False,
                                                     #pin_memory=True,
                                                     **dataloaders_param_val,
                                                     sampler=SubsetRandomSampler(val_idx))
                # print n-th fold
                print("\n\n********** Fold ", fold+1, "**************")
                # the training and validation loop
                if parallel_tpu == False:
                    valloss, valacc = self._training_loops(n_epochs, dl_tr,
                                                       dl_val, lr_scheduler, scheduler,
                                                       prof, check_optuna, search_metric,
                                                       trial, writer_tag + "/fold = " + str(fold+1))
                else:
                    valloss, valacc = self.parallel_tpu_training_loops(n_epochs, dl_tr,
                                                       dl_val, optimizers_param, lr_scheduler,
                                                       scheduler,
                                                       prof, check_optuna, search_metric,
                                                       trial)
                
                mean_val_loss.append(valloss)
                mean_val_acc.append(valacc)
                # mean of the validation and loss accuracies over folds
            valloss = np.mean(mean_val_loss)
            valacc = np.mean(mean_val_acc)


        else:
            self.reset_model()
            self.optimizer = optimizer(self.model.parameters(), **optimizers_param)
            if not lr_scheduler is None:
                scheduler = lr_scheduler(self.optimizer, **scheduler_params)

            if parallel_tpu == False:
                valloss, valacc = self._training_loops(n_epochs, dl_tr,
                                                   dl_val, lr_scheduler, scheduler,
                                                   prof, check_optuna, search_metric,
                                                   trial, writer_tag)
            else:
                valloss, valacc = self.parallel_tpu_training_loops(n_epochs, dl_tr,
                                                   dl_val, optimizers_param, lr_scheduler,
                                                   scheduler,
                                                   prof, check_optuna, search_metric,
                                                   trial)
        try:
            self.writer.flush()
        except AttributeError:
            pass
        # put the mean of the cross_val
        
        return valloss, valacc
        
    def _training_loops(self, n_epochs, dl_tr,
                        dl_val, lr_scheduler, scheduler,
                        prof, check_optuna, search_metric,
                        trial, writer_tag=""):
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
            writer_tag (str):
                the tensorboard writer tag

        Returns:
            (float, float):
                the validation loss and validation accuracy
        """

        valloss, valacc = 0, 0
        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.val_epoch = t
            self.train_epoch = t
            self._train_loop(dl_tr, writer_tag)
            valloss, valacc = self._val_loop(dl_val, writer_tag)
            
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


    def parallel_tpu_training_loops(self, n_epochs, dl_tr_old,
                                    dl_val_old, optimizers_param,
                                    lr_scheduler, scheduler,
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
        self.val_loss = 0
        self.val_acc = 0
        warnings.warn("The tensorboard writer is ignored " +
                      "for multi TPU processing")
        def map_fun_custom(index, flags):
            """map function for multi-processing"""
            device = xm.xla_device()

            print("uploading model to TPU")
            model2 = self.model.to(device)

            # initialize optimizer
            optimizer_class = type(self.optimizer)
            optimizer = optimizer_class(model2.parameters(),**optimizers_param)

            # define training and validation
            # distributed samplers and update
            # the dataloaders
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dl_tr_old.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
            
            dl_tr = torch.utils.data.DataLoader(
                dl_tr_old.dataset,
                num_workers=dl_tr_old.num_workers,
                batch_size=dl_tr_old.batch_size,
                sampler=train_sampler,
                drop_last=dl_tr_old.drop_last
            )
          
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dl_val_old.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)
            
            dl_val = torch.utils.data.DataLoader(
                dl_val_old.dataset,
                num_workers=dl_val_old.num_workers,
                batch_size=dl_val_old.batch_size,
                sampler=val_sampler,
                drop_last=dl_val_old.drop_last
            )


            #train loop
            for t in range(n_epochs):
                model2.train()
                para_train_loader = pl.ParallelLoader(dl_tr,
                                                      [device]).per_device_loader(device)
                print(f"Epoch {t+1}\n-------------------------------")
                self.val_epoch = t
                self.train_epoch = t

                # train batch loop
                size = len(dl_tr.dataset)
                steps = len(dl_tr)
                loss = 0    # arbitrary starting value to avoid nan loss
                correct = 0
                tik = time.time()
                # for batch, (X, y) in enumerate(self.dataloaders[0]):
                for batch, (X, y) in enumerate(para_train_loader):
                    # Compute prediction and loss
                    pred = model2(X)
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    loss = self.loss_fn(pred, y)
                    # Save to tensorboard
                    #self.writer.add_scalar("Parallel" + "/Loss/train",
                    #                       loss.cpu(),
                    #                       self.train_epoch*dl_tr.batch_size + batch)
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()

                    xm.optimizer_step(self.optimizer)

                #self.writer.flush()
                # train accuracy:
                correct /= len(dl_tr)*dl_tr.batch_size
                print("Train accuracy at epoch ",t," : ", correct)
                print("\nTime taken for this epoch: {}s".format(round(time.time()-tik), 2))

                # evaluation
                model2.eval()
                
                # size = len(self.dataloaders[1].dataset)
                size = len(dl_val.dataset)
                loss, correct = 0, 0
                class_label = []
                class_probs = []

                pred = 0
                para_valid_loader = pl.ParallelLoader(dl_val,
                                                      [device]).per_device_loader(device)
                with torch.no_grad():
                    # per batch!!
                    for X, y in para_valid_loader:
                        pred = model2(X)
                        class_probs_batch = [F.softmax(el, dim=0)
                                             for el in pred]
                        class_probs.append(class_probs_batch)
                        loss += self.loss_fn(pred, y).item()
                        correct += (pred.argmax(1) ==
                                    y).type(torch.float).sum().item()
                        class_label.append(y)
                    # add data to tensorboard
                    #self._add_pr_curve_tb(pred, class_label, class_probs, "validation")

                #self.writer.add_scalar("Parallel " + "/Accuracy/validation", correct, self.val_epoch)
                print(f"Validation results: \n Accuracy: {(100*correct):>0.1f}%, \
                        Avg loss: {loss:>8f} \n")

                #self.writer.flush()
              
                if not lr_scheduler is None:
                    scheduler.step()
                if not prof is None:
                    prof.step()

                if check_optuna:
                    if search_metric == "loss":
                        trial.report(loss, t)
                    else:
                        trial.report(correct, t)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            self.val_loss += loss
            self.val_acc += correct*100

        flags = {}
        self.val_acc /= len(dl_val_old)*dl_val_old.batch_size
        self.val_loss /= len(dl_val_old)
        xmp.spawn(map_fun_custom, args=(flags,), nprocs=8, start_method='fork')
        return self.val_loss, self.val_acc

    def evaluate_classification(self, num_class=None, dl=None):
        """Method to evaluate the performance of the model.
        
        Args:
            num_class (int):
                number of classes
            dl (torch.DataLoader, default None):
                the Dataloader to evaluate. If ``None``,
                we use the training dataloader in ``self``
                
        Returns:
            (float, float, 2darray):
                the accuracy, loss and confusion matrix.
        """
        if dl is None:
            dl = self.dataloaders[0]
        class_probs = []
        class_label = []
        loss = 0
        correct = 0
        confusion_matrix = np.zeros((num_class, num_class))
        self.model.eval()
        with torch.no_grad():
            for X, y in tqdm(dl):
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
                for t, p in zip(y.view(-1), pred.argmax(1).view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        correct /= len(dl)*dl.batch_size
        loss /= len(dl)
        return 100*correct, loss, confusion_matrix


import torch.nn.functional as F
import torch
import numpy as np
import copy
import time
import warnings
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.data import PreprocessText

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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
        self.initial_model = copy.deepcopy(self.model)
        # assert len(dataloaders) == 2 or len(dataloaders) == 3
        # self.dataloaders = dataloaders  # train and test
        assert len(dataloaders) > 0 and len(dataloaders) < 4, "Length of dataloaders must be 1, 2, or 3"
        self.dataloaders = dataloaders  # train and test
        self.train_epoch = 0
        self.val_epoch = 0

        # else:
        self.loss_fn = loss_fn
        # integrate tensorboard
        self.writer = writer
        
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
            #print("loss fnc: ", loss)
            # Save to tensorboard
            self.writer.add_scalar(writer_tag + "/Loss/train", loss, self.train_epoch*dl_tr.batch_size + batch)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print(list(self.model.parameters()))
            if batch % 1 == 0:
                t_loss = loss.item()
                print("Training loss: ", t_loss, " [",batch+1,"/",
                      steps,"]                     ", end='\r')

        # accuracy:
        correct /= size
        print("\nTime taken for this epoch: {}s".format(round(time.time()-tik), 2))
        self.writer.flush()

        return loss, correct

    def _val_loop(self, dl_val, writer_tag=""):
        """private method to run a single validation
        loop
        """

        # size = len(self.dataloaders[1].dataset)
        size = len(dl_val.dataset)
        val_loss, correct = 0, 0
        class_label = []
        class_probs = []
        self.model.eval()
        pred = 0

        # for X, y in self.dataloaders[1]:
        for X, y in dl_val:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = self.model(X)
            class_probs_batch = [F.softmax(el, dim=0)
                                 for el in pred]
            class_probs.append(class_probs_batch)
            val_loss += self.loss_fn(pred, y).item()
            correct += (pred.argmax(1) ==
                        y).type(torch.float).sum().item()
            class_label.append(y)
        # add data to tensorboard
        val_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        val_label = torch.cat(class_label)

        for class_index in range(len(pred[0])):
            tensorboard_truth = val_label == class_index
            tensorboard_probs = val_probs[:, class_index]
            self.writer.add_pr_curve(writer_tag+str(class_index),
                                     tensorboard_truth,
                                     tensorboard_probs,
                                     global_step=0)
        self.writer.flush()

        # accuracy
        val_loss /= size
        correct /= size

        self.writer.add_scalar(writer_tag + "/Accuracy/validation", correct, self.val_epoch)
        print(f"Validation results: \n Accuracy: {(100*correct):>0.1f}%, \
                Avg loss: {val_loss:>8f} \n")

        self.writer.flush()

        return val_loss, 100*correct

    def _test_loop(self, dl_test, writer_tag=""):
        """private method to run a single test
        loop
        """

        # size = len(self.dataloaders[1].dataset)
        size = len(dl_test.dataset)
        test_loss, correct = 0, 0
        class_label = []
        class_probs = []
        self.model.eval()
        pred = 0

        # for X, y in self.dataloaders[1]:
        for X, y in dl_test:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = self.model(X)
            class_probs_batch = [F.softmax(el, dim=0)
                                 for el in pred]
            class_probs.append(class_probs_batch)
            test_loss += self.loss_fn(pred, y).item()
            correct += (pred.argmax(1) ==
                        y).type(torch.float).sum().item()
            class_label.append(y)
        # add data to tensorboard
        test_probs = torch.cat([torch.stack(batch) for batch in
                                class_probs])
        test_label = torch.cat(class_label)

        for class_index in range(len(pred[0])):
            tensorboard_truth = test_label == class_index
            tensorboard_probs = test_probs[:, class_index]
            self.writer.add_pr_curve(writer_tag+str(class_index),
                                     tensorboard_truth,
                                     tensorboard_probs,
                                     global_step=0)
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
              scheduler_params=None):
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
            
        Returns:
            (float, float):
                the validation loss and accuracy
                if there is cross validation, the validation data loader
                is ignored. On the other hand, if there `cross_validation = False`
                then the test loss and accuracy is returned.
        """

        if optimizers_param is None:
            optimizers_param = {"lr":0.001}
        dl_tr = self.dataloaders[0]
        if dataloaders_param is None:
            dataloaders_param = {"batch_size":dl_tr.batch_size}
        if scheduler_params is None:
            scheduler_params = {}
        # CV folds
        k_folds = 5
        
        # profiling
        if not cross_validation:
            active = n_epochs-2
        else:
            active = k_folds*(n_epochs-2)
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
        
        #print("here:",len(dl_tr))
        if len(self.dataloaders) == 3:
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
                # initialise data loaders
                if len(self.dataloaders) == 3:
                    warnings.warn("Validation set is ignored in automatic Cross Validation")
                dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    **dataloaders_param,
                                                    sampler=SubsetRandomSampler(tr_idx))
                dl_val = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     **dataloaders_param,
                                                     sampler=SubsetRandomSampler(val_idx))
                # print n-th fold
                print("\n\n********** Fold ", fold+1, "**************")


                # the training and validation
                for t in range(n_epochs):
                    print(f"Epoch {t+1}\n-------------------------------")
                    self.val_epoch = t
                    self.train_epoch = t
                    self._train_loop(dl_tr, "train fold"+str(1+fold))
                    valloss, valacc = self._val_loop(dl_val, "validation fold"+str(1+fold))
                if not lr_scheduler is None:
                    scheduler.step()
                try:
                    prof.step()
                except UnboundLocalError:
                    pass
                
                mean_val_loss.append(valloss)
                mean_val_acc.append(valacc)
                # mean of the validation and loss accuracies over folds
                valloss = np.mean(mean_val_loss)
                valacc = np.mean(mean_val_acc)

        else:
            if not dataloaders_param == {}:
                dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    sampler=range(len(dl_tr)*dl_tr.batch_size),
                                                    **dataloaders_param)
            self.reset_model()
            self.optimizer = optimizer(self.model.parameters(), **optimizers_param)
            if not lr_scheduler is None:
                scheduler = lr_scheduler(self.optimizer, **scheduler_params)

            for t in range(n_epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                self._train_loop(dl_tr, "train")
                if len(self.dataloaders) == 3:
                    valloss, valacc = self._val_loop(dl_val, "validation")
                    self.val_epoch = t
                self.train_epoch = t
                #print(self.optimizer.param_groups[0]["lr"])
                if not lr_scheduler is None:
                    scheduler.step()
                try:
                    prof.step()
                except UnboundLocalError:
                    pass
                    

            # test the results
            if len(self.dataloaders) == 2:
                valloss, valacc = self._test_loop(self.dataloaders[1], "test")
            
        self.writer.flush()
        # put the mean of the cross_val
        
        return valloss, valacc

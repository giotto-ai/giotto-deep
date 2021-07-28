import torch.nn.functional as F
import torch
import time
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler

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
        dataloader (utils.DataLoader)
        loss_fn (Callables)
        wirter (tensorboard SummaryWriter)

    """

    # def __init__(self, model, dataloaders, loss_fn, writer,
    # hyperparams_search = False, search_metric = "accuracy", n_trials = 10):
    def __init__(self, model, dataloaders, loss_fn, writer):
        self.model = model.to(DEVICE)
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

    def reset_epoch(self):
        """method to reset global training and validation
        epoch count
        """

        self.train_epoch = 0
        self.val_epoch = 0

    def _train_loop(self, dl_tr, writer_tag=""):
        """private method to run a single training
        loop
        """

        steps = len(dl_tr)

        tik = time.time()
        # for batch, (X, y) in enumerate(self.dataloaders[0]):
        for batch, (X, y) in enumerate(dl_tr):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Save to tensorboard
            self.writer.add_scalar(writer_tag + "/Loss/train", loss, self.train_epoch)
            self.train_epoch = self.train_epoch + 1
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 1 == 0:
                loss, _ = loss.item(), (batch+1) * len(X)
                print(f"Training loss: {loss:>7f}  [{batch+1:>2d}/{steps:>2d}]", end="\r")

        print("\nTime taken for this epoch: {}s".format(round(time.time()-tik), 2))
        self.writer.flush()

        return loss

    def _val_loop(self, dl_val, writer_tag=""):
        """private method to run a single validation
        loop
        """

        # size = len(self.dataloaders[1].dataset)
        size = len(dl_val.dataset)
        val_loss, correct = 0, 0
        class_label = []
        class_probs = []
        with torch.no_grad():
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
                self.writer.add_pr_curve(str(class_index),
                                         tensorboard_truth,
                                         tensorboard_probs,
                                         global_step=0)
        self.writer.flush()

        val_loss /= size
        correct /= size

        self.writer.add_scalar(writer_tag + "/Accuracy/validation", correct, self.val_epoch)
        print(f"Validation results: \n Accuracy: {(100*correct):>0.1f}%, \
                Avg loss: {val_loss:>8f} \n")

        self.writer.flush()

        return 100*correct

    def _test_loop(self, dl_test, writer_tag=""):
        """private method to run a single test
        loop
        """

        # size = len(self.dataloaders[1].dataset)
        size = len(dl_test.dataset)
        test_loss, correct = 0, 0
        class_label = []
        class_probs = []
        with torch.no_grad():
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
                self.writer.add_pr_curve(str(class_index),
                                         tensorboard_truth,
                                         tensorboard_probs,
                                         global_step=0)
        self.writer.flush()

        test_loss /= size
        correct /= size
        print(f"Test results: \n Accuracy: {(100*correct):>0.1f}%, \
                Avg loss: {test_loss:>8f} \n")

        return 100*correct

    def train(self, optimizer, n_epochs=10, cross_validation=False, batch_size=512, **kwargs):
        """Function to run the trianing cycles.

        Args:
            optimiser (torch.optim)
            n_epochs (int)
            cross_validation (bool)
            batch_size (int)
        """

        self.optimizer = optimizer(self.model.parameters(), **kwargs)

        if len(self.dataloaders) == 3:
            dl_tr = self.dataloaders[0]
            dl_val = self.dataloaders[1]

        k_folds = 5
        data_idx = list(range(len(self.dataloaders[0])*batch_size))

        # print(folds)
        fold = KFold(k_folds, shuffle=False)

        # print(self.dataloaders[0].dataset)
        for fold, (tr_idx, val_idx) in enumerate(fold.split(data_idx)):
            if len(self.dataloaders) == 1 or len(self.dataloaders) == 2:
                dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                    shuffle=False, batch_size=batch_size, sampler=SubsetRandomSampler(tr_idx))
                dl_val = torch.utils.data.DataLoader(self.dataloaders[0].dataset,
                                                     shuffle=False, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

            if cross_validation and (len(self.dataloaders) == 1 or len(self.dataloaders) == 2):
                print("\n\n********** Fold ", fold+1, "**************")

            print("TOTAL EPOCHS ", n_epochs)
            for t in range(n_epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                self._train_loop(dl_tr)
                self.val_epoch = t+1
                self._val_loop(dl_val)

            if len(self.dataloaders) == 2:
                self._test_loop(self.dataloaders[1])
            elif len(self.dataloaders) == 3:
                self._test_loop(self.dataloaders[2])
                break

            if not cross_validation:
                break

        self.writer.close()
        print("Done!")

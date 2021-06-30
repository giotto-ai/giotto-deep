import torch.nn.functional as F
import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import time

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

    def __init__(self, model, dataloaders, loss_fn, writer, hyperparams_search = False, search_metric = "accuracy", n_trials = 10):
        self.model = model.to(DEVICE)
        assert len(dataloaders) == 2 or len(dataloaders) == 3
        self.dataloaders = dataloaders  # train and test

        self.loss_fn = loss_fn
        # integrate tensorboard
        self.writer = writer

        self.hyperparams_search = hyperparams_search
        self.search_metric = search_metric
        self.n_trials = n_trials

    def _train_loop(self):
        """private method to run a single training
        loop
        """
        size = len(self.dataloaders[0].dataset)
        steps = len(self.dataloaders[0])

        tik = time.time()
        for batch, (X, y) in enumerate(self.dataloaders[0]):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Save to tensorboard
            self.writer.add_scalar("Loss/train", loss, batch)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 1 == 0:
                loss, current = loss.item(), (batch+1) * len(X)
                print(f"loss: {loss:>7f}  [{batch+1:>2d}/{steps:>2d}]", end="\r")
        
        print("Time taken for this epoch: {}s".format(round(time.time()-tik), 2))
        self.writer.flush()

        return loss

    def _test_loop(self):
        """private method to run a single test
        loop
        """
        size = len(self.dataloaders[1].dataset)
        test_loss, correct = 0, 0
        class_label = []
        class_probs = []
        with torch.no_grad():
            pred = 0
            
            for X, y in self.dataloaders[1]:
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
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, \
                Avg loss: {test_loss:>8f} \n")
        
        return 100*correct
    
    def objective(self, trial, optimizer, n_epochs=10, **kwargs):
        if isinstance(optimizer, list):
            optimizer = trial.suggest_categorical("optimizer", optimizer)
        # else:
        #     optimizer_name = optimizer

        # self.optimizer = optimizer(self.model.parameters(), **kwargs)
        if isinstance(kwargs["lr"], list):
            kwargs["lr"] = trial.suggest_float("lr", kwargs["lr"][0], kwargs["lr"][1], log=True)

        self.optimizer = optimizer(self.model.parameters(), **kwargs)

        

        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            loss = self._train_loop()
            accuracy = self._test_loop()

            if self.search_metric == "loss":
                trial.report(loss, t)
            else:
                trial.report(accuracy, t)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self.writer.close()
        print("Done!")

        if self.search_metric == "loss":
            return loss
        else:
           return accuracy
        

    def train(self, optimizer, n_epochs=10, **kwargs):
        """Function to run the trianing cycles.

        Args:
            optimiser (torch.optim)
            n_epochs (int)
        """

        if self.hyperparams_search:
            if self.search_metric == "loss":
                study = optuna.create_study(direction="minimize")
            else:
                study = optuna.create_study(direction="maximize")

            study.optimize(lambda trial: self.objective(trial, optimizer, n_epochs, **kwargs), n_trials=self.n_trials, timeout=600)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

        else:
            if isinstance(optimizer, list):
                optimizer = optimizer[0]

            if isinstance(kwargs["lr"], list):
                kwargs["lr"] = kwargs["lr"][0]
            
            print(kwargs)
            print(optimizer)
            self.optimizer = optimizer(self.model.parameters(), **kwargs)
            print("TOTAL EPOCHS ", n_epochs)
            for t in range(n_epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                self._train_loop()
                self._test_loop()
            self.writer.close()
            print("Done!")
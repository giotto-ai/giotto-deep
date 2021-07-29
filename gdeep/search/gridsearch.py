import torch
import optuna
from optuna.trial import TrialState
from gdeep.pipeline import Pipeline
from gdeep.search.benchmark import Benchmark
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class Gridsearch(Pipeline, Benchmark):
    """This is the generic class that allows
    the user to perform gridsearch over several
    parameters such as learning rate, optimizer.

    Args:
        obj (either Pipeline or Benchmark object):
        search_metric (string)
        n_trials (int)

    """

    def __init__(self, obj, search_metric="loss", n_trials=10):
        self.pipe = obj
        if (isinstance(obj, Pipeline)):
            super(Gridsearch, self).__init__(obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            # Pipeline.__init__(self, obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            self.pipe = True
        elif (isinstance(obj, Benchmark)):
            Benchmark.__init__(self, obj.models_dicts, obj.dataloaders_dicts, obj.loss_fn, obj.writer)
            self.bench = obj
            self.pipe = False

        self.search_metric = search_metric
        self.n_trials = n_trials
        self.val_epoch = 0

    # def __init__(self, bench, search_metric="loss", n_trials=10, temp=True):
    #     self.bench = bench
    #     super(Gridsearch, self).__init__(bench.models_dicts, bench.dataloaders_dicts, bench.loss_fn, bench.writer)
    #     self.search_metric = search_metric
    #     self.n_trials = n_trials
    #     self.val_epoch = 0
    #     self.pipe = False

    def objective(self, trial, optimizer, n_epochs=10, batch_size=512, writer_string="", **kwargs):
        """default callback function for optuna's study
        """

        if isinstance(optimizer, list):
            optimizer = trial.suggest_categorical("optimizer", optimizer)
        # else:
        #     optimizer_name = optimizer

        # self.optimizer = optimizer(self.model.parameters(), **kwargs)
        if isinstance(kwargs["lr"], list):
            kwargs["lr"] = trial.suggest_float("lr", kwargs["lr"][0], kwargs["lr"][1], log=True)

        self.optimizer = optimizer(self.model.parameters(), **kwargs)

        if len(self.dataloaders) == 3:
            dl_tr = self.dataloaders[0]
            dl_val = self.dataloaders[1]

        k_folds = 5
        data_idx = list(range(len(self.dataloaders[0])*batch_size))

        fold = KFold(k_folds, shuffle=False)

        Pipeline.reset_epoch(self)

        for fold, (tr_idx, val_idx) in enumerate(fold.split(data_idx)):
            if len(self.dataloaders) == 1 or len(self.dataloaders) == 2:
                dl_tr = torch.utils.data.DataLoader(self.dataloaders[0].dataset, shuffle=False,
                                                    batch_size=batch_size, sampler=SubsetRandomSampler(tr_idx))
                dl_val = torch.utils.data.DataLoader(self.dataloaders[0].dataset, shuffle=False,
                                                     batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
            break

        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            loss = super(Gridsearch, self)._train_loop(dl_tr,
                                        writer_string + ", Gridsearch trial: " + str(trial.number) + ", " + str(trial.params))
            self.val_epoch = t
            accuracy = super(Gridsearch, self)._val_loop(dl_val, writer_string +
                                          ", Gridsearch trial: " + str(trial.number) + ", " + str(trial.params))

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

    def start(self, optimizer, n_epochs=10, batch_size=512, **kwargs):
        """method to be called when starting the gridsearch
        """
        if self.pipe:
            if self.search_metric == "loss":
                study = optuna.create_study(direction="minimize")
            else:
                study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, optimizer, n_epochs,
                                                        batch_size, **kwargs), n_trials=self.n_trials, timeout=None)

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
            for dataloaders in self.bench.dataloaders_dicts:
                for model in self.bench.models_dicts:
                    print("*"*40)
                    print("Performing Gridsearch on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))

                    writer_string = "Dataset: " + dataloaders["name"] + " | Model: " + model["name"]

                    if self.search_metric == "loss":
                        study = optuna.create_study(direction="minimize")
                    else:
                        study = optuna.create_study(direction="maximize")

                    super(Gridsearch, self).__init__(model["model"], dataloaders["dataloaders"], self.bench.loss_fn, self.bench.writer)

                    study.optimize(lambda trial: self.objective(trial, optimizer, n_epochs, batch_size, writer_string,
                                                                **kwargs), n_trials=self.n_trials, timeout=None)

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

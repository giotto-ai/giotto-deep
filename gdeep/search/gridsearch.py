import torch
import optuna
import pandas as pd
import numpy as np
from optuna.trial import TrialState
from gdeep.pipeline import Pipeline
from gdeep.search.benchmark import Benchmark
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.visualisation import plotly2tensor
from torch.optim import *
import plotly.express as px


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
        self.pipe = None
        self.obj = obj
        self.bench = None
        self.study = None
        self.metric = search_metric
        self.list_res = []
        if (isinstance(obj, Pipeline)):
            super(Gridsearch, self).__init__(obj.model, 
                                             obj.dataloaders, 
                                             obj.loss_fn, 
                                             obj.writer)
            # Pipeline.__init__(self, obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            self.pipe = True
        elif (isinstance(obj, Benchmark)):
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

    def objective(self, trial, optimizers, n_epochs=10, 
                  batch_size=512, writer_string="", 
                  **kwargs):
        """default callback function for optuna's study
        """

        if isinstance(optimizers, list) or isinstance(optimizers, tuple):
            optimizers_names = list(map(lambda x: x.__name__, optimizers))
            optimizer = eval(trial.suggest_categorical("optimizer", optimizers_names))
        else:
            optimizer = optimizers

        # self.optimizer = optimizer(self.model.parameters(), **kwargs)
        if isinstance(kwargs["lr"], list) or isinstance(kwargs["lr"], tuple):
            kwargs["lr"] = trial.suggest_float("lr", kwargs["lr"][0], kwargs["lr"][1], log=True)
        kwargs_optim = {k:kwargs[k] for k in ('n_epochs','batch_size') if k in kwargs}

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
                                                       writer_string + 
                                                       ", Gridsearch trial: " + 
                                                       str(trial.number) + 
                                                       ", " + 
                                                       str(trial.params))
            self.val_epoch = t
            accuracy = super(Gridsearch, self)._val_loop(dl_val, 
                                                         writer_string +
                                                         ", Gridsearch trial: " + 
                                                         str(trial.number) + 
                                                         ", " + 
                                                         str(trial.params))

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
                self.study = optuna.create_study(direction="minimize")
            else:
                self.study = optuna.create_study(direction="maximize")
            self.study.optimize(lambda trial: self.objective(trial, 
                                                        optimizer, 
                                                        n_epochs,
                                                        batch_size,
                                                        **kwargs), 
                           n_trials=self.n_trials, 
                           timeout=None)
            self.results()

        else:
            for dataloaders in self.bench.dataloaders_dicts:
                for model in self.bench.models_dicts:
                    print("*"*40)
                    print("Performing Gridsearch on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))

                    writer_string = "Dataset: " + dataloaders["name"] + " | Model: " + model["name"]

                    if self.search_metric == "loss":
                        self.study = optuna.create_study(direction="minimize")
                    else:
                        self.study = optuna.create_study(direction="maximize")

                    super(Gridsearch, self).__init__(model["model"], 
                                                     dataloaders["dataloaders"], 
                                                     self.bench.loss_fn, 
                                                     self.bench.writer)

                    self.study.optimize(lambda trial: self.objective(trial, 
                                                                     optimizer, 
                                                                     n_epochs, 
                                                                     batch_size, 
                                                                     writer_string,
                                                                     **kwargs), 
                                        n_trials=self.n_trials, 
                                        timeout=None)
                    self.results(model_name = model["name"], dataset_name = dataloaders["name"])

                        
                        
    def results(self, model_name = "model", dataset_name = "dataset"):
        """This class returns the dataframe with all the results of
        the gridsearch. It also saves the figures in the writer."""
        
        trials = self.study.trials
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("Number of finished trials: ", len(trials))
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial_best = self.study.best_trial

        print("Metric Value for best trial: ", trial_best.value)
        
        for trial in trials:
            temp_list = []
            for val in trial.params.values():
                temp_list.append(val)
                self.list_res.append([model_name, dataset_name] + temp_list + [trial.value])

        
   
        # already present in tennsorboard
        df_res = pd.DataFrame(self.list_res, columns=["model", "dataset"] +
                              list(trial_best.params.keys())+["Metric value"])

        fig = px.parallel_coordinates(df_res, color="Metric value", labels=df_res.columns,
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)
        
        # correlations of numercal coefficients
        
        list_of_arrays = []
        labels = []
        for col in df_res.columns:
            vals = df_res[col].values
            if vals.dtype in [np.float16, np.float64, 
                                 np.float32,
                                 np.int32, np.int64, 
                                 np.int16]:
                list_of_arrays.append(vals)
                labels.append(col)
            
        corr = np.corrcoef(np.array(list_of_arrays))
        
        fig2 = px.imshow(corr,
                labels=dict(x="Parameters", 
                            y="Parameters", 
                            color="Correlation"),
                x=labels,
                y=labels
               )
        fig2.update_xaxes(side="top")
       
        #img1 = plotly2tensor(fig)
        img2 = plotly2tensor(fig2)
            
        #self.obj.writer.add_images("Gridsearch parallel plot: " + model_name + " " + dataset_name,
        #                            img1, dataformats="HWC")
        self.obj.writer.add_images("Gridsearch correlation: " + model_name + " " + dataset_name,
                                    img2, dataformats="HWC")
        for i in range(len(df_res)):
            self.obj.writer.add_hparams(dict(df_res.iloc[i][:-1]), dict({self.metric: df_res.iloc[i][-1]}))
        
        
        self.obj.writer.flush()
        
        return df_res

import torch
import optuna
import pandas as pd
import numpy as np
from gdeep.utility import _are_compatible
from optuna.trial import TrialState
from gdeep.pipeline import Pipeline
from gdeep.search import Benchmark
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.visualisation import plotly2tensor
from torch.optim import *
import plotly.express as px


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class Gridsearch(Pipeline):
    """This is the generic class that allows
    the user to perform gridsearch over several
    parameters such as learning rate, optimizer.

    Args:
        obj (either Pipeline or Benchmark object):
        search_metric (string): either 'loss' or 'accuracy'
        n_trials (int): number of total gridsearch trials

    """

    def __init__(self, obj, search_metric="loss", n_trials=10):
        self.is_pipe = None
        self.obj = obj
        self.bench = obj
        self.study = None
        self.metric = search_metric
        self.list_res = []
        self.df_res = None
        if (isinstance(obj, Pipeline)):
            super().__init__(self.obj.model,
                             self.obj.dataloaders,
                             self.obj.loss_fn,
                             self.obj.writer)
            # Pipeline.__init__(self, obj.model, obj.dataloaders, obj.loss_fn, obj.writer)
            self.is_pipe = True
        elif (isinstance(obj, Benchmark)):
            #super().__init__(obj.models_dicts,
            #                 obj.dataloaders_dicts,
            #                 obj.loss_fn,
            #                 obj.writer)
            self.is_pipe = False

        self.search_metric = search_metric
        self.n_trials = n_trials
        self.val_epoch = 0
        self.train_epoch = 0


    def _objective(self, trial,
                   optimizers,
                   n_epochs,
                   optimizers_params,
                   dataloaders_params,
                   models_hyperparams,
                   writer_tag="",
                   **kwargs):
        """default callback function for optuna's study
        
        Args:
            trial (optuna.trial): the independent variable
            pipel (Pipeline): a Pipeline
            optimizers (list of torch.optim): list of torch optimizers
            n_epochs (int): number of training epochs
            batch_size (int): batch size for the training
            writer_tag (string): tag to prepend to the ouput
                on tensorboard
        """

        
        optimizers_names = list(map(lambda x: x.__name__, optimizers))
        optimizer = eval(trial.suggest_categorical("optimizer", optimizers_names))
        
        # generate all the hyperparameters
        optimizers_param = self.suggest_params(trial, optimizers_params)
        #print(optimizers_param)
        dataloaders_param = self.suggest_params(trial, dataloaders_params)
        #print(dataloaders_param)
        models_hyperparam = self.suggest_params(trial, models_hyperparams)
        #print(models_hyperparam)
        #print(self.model)
        # create a new model instance
        try:
            new_model = type(self.model)(**models_hyperparam)
        except TypeError:
            new_model = self.model

        #print(new_model)
        new_pipe = Pipeline(new_model, self.dataloaders, self.loss_fn, self.writer)
        for t in range(n_epochs):
            loss, accuracy = new_pipe.train(optimizer, 1, False,
                                            dataloaders_param,
                                            optimizers_param)

            if self.search_metric == "loss":
                trial.report(loss, t)
            else:
                trial.report(accuracy, t)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self.writer.flush()
        print("Done!")
        del(new_pipe)
        del(new_model)
        if self.search_metric == "loss":
            return loss
        else:
            return accuracy

    def start(self,
              optimizers,
              n_epochs,
              optimizers_params = None,
              dataloaders_params = None,
              models_hyperparams = None,
              **kwargs):
        """method to be called when starting the gridsearch
        
        Args:
            trial (optuna.trial): the independent variable
            optimizers (list of torch.optim): list of torch optimizers
            n_epochhs (int): number of training epochs
            optimizers_params (dict): number of training epochs
            dataloaders_params (int): batch size for the training
            writer_tag (string): tag to prepend to the ouput
                on tensorboard
        """
        if self.is_pipe:
            if self.search_metric == "loss":
                self.study = optuna.create_study(direction="minimize")
            else:
                self.study = optuna.create_study(direction="maximize")
            
            self.study.optimize(lambda tr: self._objective(tr,
                                                           optimizers,
                                                           n_epochs,
                                                           optimizers_params,
                                                           dataloaders_params,
                                                           models_hyperparams,
                                                           writer_tag = "model",
                                                           **kwargs),
                                n_trials=self.n_trials,
                                timeout=None)
            self.results()

        else:
            for dataloaders in self.bench.dataloaders_dicts:
                for model in self.bench.models_dicts:
                    if _are_compatible(model, dataloaders):
                        print("*"*40)
                        print("Performing Gridsearch on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))

                        writer_tag = "Dataset: " + dataloaders["name"] + " | Model: " + model["name"]

                        if self.search_metric == "loss":
                            self.study = optuna.create_study(direction="minimize")
                        else:
                            self.study = optuna.create_study(direction="maximize")

                        super().__init__(model["model"],
                                         dataloaders["dataloaders"],
                                         self.bench.loss_fn,
                                         self.bench.writer)

                        self.study.optimize(lambda tr: self._objective(tr,
                                                                       optimizers,
                                                                       n_epochs,
                                                                       optimizers_params,
                                                                       dataloaders_params,
                                                                       models_hyperparams,
                                                                       writer_tag,
                                                                       **kwargs),
                                            n_trials=self.n_trials,
                                            timeout=None)
                        self.results(model_name = model["name"],
                                     dataset_name = dataloaders["name"])
        self.store_to_tensorboard()

                        
                        
    def results(self, model_name = "model", dataset_name = "dataset"):
        """This class returns the dataframe with all the results of
        the gridsearch. It also saves the figures in the writer."""
        
        trials = self.study.trials
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))
        try:
            print("Best trial:")
            trial_best = self.study.best_trial

            print("Metric Value for best trial: ", trial_best.value)
        except ValueError:
            pass
        
        for tria in trials:
            temp_list = []
            for val in tria.params.values():
                temp_list.append(val)
                #print("value here:", tria.value)
            self.list_res.append([model_name, dataset_name] + temp_list + [tria.value])
        # already present in tensorboard

        self.df_res = pd.DataFrame(self.list_res, columns=["model", "dataset"] +
                              list(trial_best.params.keys())+[self.metric])

        fig = px.parallel_coordinates(self.df_res, color=self.metric, labels=self.df_res.columns,
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)
        
        # correlations of numercal coefficients
        
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
        
        fig2 = px.imshow(corr,
                labels=dict(x="Parameters", 
                            y="Parameters", 
                            color="Correlation"),
                x=labels,
                y=labels
               )
        fig2.update_xaxes(side="top")
        fig2.show()
       
        #img1 = plotly2tensor(fig)
        img2 = plotly2tensor(fig2)
            
        #self.obj.writer.add_images("Gridsearch parallel plot: " + model_name + " " + dataset_name,
        #                            img1, dataformats="HWC")
        self.writer.add_images("Gridsearch correlation: " + model_name + " " + dataset_name,
                                    img2, dataformats="HWC")
        self.writer.flush()
        
        return self.df_res
        
    def store_to_tensorboard(self):
        for i in range(len(self.df_res)):
            dictio = {k:(int(v) if isinstance(v, np.int64) else v) for k,v in dict(self.df_res.iloc[i][:-1]).items()}
            self.writer.add_hparams(dictio,
                                    {self.df_res.columns[-1]: self.df_res.iloc[i][-1]})
        
        self.writer.flush()
        
        return self.df_res

    @staticmethod
    def suggest_params(trial, optimizers_params):
        optimizers_param_temp = {k:trial.suggest_float(k,*v) for k,v in
                                optimizers_params.items() if (type(v) is list or type(v) is tuple)
                                and (type(v[0]) is float or type(v[1]) is float)}
        optimizers_param_temp2 = {k:trial.suggest_int(k,*v) for k,v in
                                 optimizers_params.items() if (type(v) is list or type(v) is tuple)
                                 and (type(v[0]) is int or type(v[1]) is int)}
        optimizers_param_temp.update(optimizers_param_temp2)
        optimizers_param = {k:trial.suggest_categorical(k, v) for k,v in
                            optimizers_params.items() if (type(v) is list or type(v) is tuple)
                            and (type(v[0]) is str or type(v[1]) is str)}
        optimizers_param.update(optimizers_param_temp)
        
        return optimizers_param

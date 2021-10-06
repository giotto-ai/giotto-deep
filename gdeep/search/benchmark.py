from gdeep.pipeline import Pipeline
from gdeep.utility import _are_compatible

class Benchmark:
    """This is the generic class that allows
    the user to perform benchmarking over different
    datasets and models.

    Args:
        model (nn.Module):
        dataloader (utils.DataLoader)
        loss_fn (Callables)
        wirter (tensorboard SummaryWriter)

    """

    def __init__(self, models_dicts, dataloaders_dicts, loss_fn, writer):
        self.models_dicts = models_dicts
        self.dataloaders_dicts = dataloaders_dicts
        self.loss_fn = loss_fn
        self.writer = writer

    def start(self, optimizer,
              n_epochs=10,
              cross_validation=False,
              optimizer_param=None,
              dataloaders_param=None,
              lr_scheduler=None,
              scheduler_params=None):
        """method to be called when starting the benchmarking
        """
        if not isinstance(self.models_dicts, list):
            raise TypeError("The provided models must be a Python list of dictionaries")

        if not isinstance(self.dataloaders_dicts, list):
            raise TypeError("The provided datasets must be a Python list of dictionaries")

        print("Benchmarking Started")
        for dataloaders in self.dataloaders_dicts:
            for model in self.models_dicts:
                if _are_compatible(model, dataloaders):
                    print("*"*40)
                    print("Training on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))
                    pipe = Pipeline(model["model"], dataloaders["dataloaders"], self.loss_fn, self.writer)
                    pipe.train(optimizer, n_epochs,
                               cross_validation,
                               optimizer_param,
                               dataloaders_param,
                               lr_scheduler,
                               scheduler_params)

from gdeep.pipeline import Pipeline


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

    def start(self, optimizer, epochs, batch_size, **kwargs):
        """method to be called when starting the benchmarking
        """
        if not isinstance(self.models_dicts, list):
            raise TypeError("The provided models must be a Python list of dictionaries")

        if not isinstance(self.dataloaders_dicts, list):
            raise TypeError("The provided datasets must be a Python list of dictionaries")

        print("Benchmarking Started")
        for dataloaders in self.dataloaders_dicts:
            for model in self.models_dicts:
                print("*"*30)
                print("Training on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))
                pipe = Pipeline(model["model"], dataloaders["dataloaders"], self.loss_fn, self.writer)
                pipe.train(optimizer, epochs, batch_size=batch_size, **kwargs)

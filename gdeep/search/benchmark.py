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
            print("Error: Provided models must be a Python list of dictionaries")

        if not isinstance(self.dataloaders_dicts, list):
            print("Error: Provided datasets must be a Python list of dictionaries")

        print("Benchmarking Started")
        for dataloaders in self.dataloaders_dicts:
            for model in self.models_dicts:
                print("*"*30)
                print("Training on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))
                pipe = Pipeline(model["model"], dataloaders["dataloaders"], self.loss_fn, self.writer)
                pipe.train(optimizer, epochs, batch_size=batch_size, lr=kwargs["lr"])

    # def benchmark_model(self, model, dataloaders_dicts, loss_fn, optimizer, epochs, learning_rate):
    #     if not isinstance(dataloaders_dicts, list):
    #         print ("Error: Provided datasets must be a Python list of dictionaries")
    #         return

    #     print("Benchmarking Started")
    #     for dataloaders in dataloaders_dicts:
    #         print("*"*30)
    #         print("Training on {}".format(dataloaders["name"]) )
    #         pipe = Pipeline(model, dataloaders["dataloaders"], loss_fn, self.writer)
    #         pipe.train(optimizer, epochs, lr=learning_rate)

    # def benchmark_data(self, models_dicts, dataloaders, loss_fn, optimizer, epochs, learning_rate):
    #     if not isinstance(models_dicts, list):
    #         print ("Error: Provided models must be a Python list of dictionaries")
    #         return

    #     print("Benchmarking Started")
    #     for model in models_dicts:
    #         print("*"*30)
    #         print("Training on {}".format(model["name"]) )
    #         pipe = Pipeline(model["model"], dataloaders,loss_fn, self.writer)
    #         pipe.train(optimizer, epochs, lr=learning_rate)
    
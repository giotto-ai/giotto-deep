import torch.nn.functional as F
import torch


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

    def __init__(self, model, dataloaders, loss_fn,
                 writer):
        self.model = model
        assert len(dataloaders) == 2 or len(dataloaders) == 3
        self.dataloaders = dataloaders  # train and test
        self.loss_fn = loss_fn
        # integrate tensorboard
        self.writer = writer

    def _train_loop(self):
        """private method to run a single training
        loop
        """
        size = len(self.dataloaders[0].dataset)
        for batch, (X, y) in enumerate(self.dataloaders[0]):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Save to tensorboard
            self.writer.add_scalar("Loss/train", loss, batch)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",
                      end="\r")
        self.writer.flush()

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

    def train(self, optimizer, n_epochs=10, **kwargs):
        """Function to run the trianing cycles.

        Args:
            optimiser (torch.optim)
            n_epochs (int)
        """
        self.optimizer = optimizer(self.model.parameters(),
                                   **kwargs)
        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self._train_loop()
            self._test_loop()
        self.writer.close()
        print("Done!")

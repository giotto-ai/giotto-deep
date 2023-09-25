import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import utils
import os
import sys
import time


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from pipeline_tool import SkippableTracing
from torch.distributed.pipeline.sync import Pipe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BenchmarkMode:
    def __init__(self, args):
        self.args = args
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_fn()

    def setup_model(self):
        if self.args.model == "CNN":
            self.model = utils.Net()
        elif self.args.model == "Resnet18":
            self.model = utils.CaltechResnet()
        elif self.args.model == "Resnet50":
            pass
        elif self.args.model == "orbit5k":
            pass
        else:
            raise ValueError("Given model is not known.")
        
        if self.args.framework == "Pipeline": 
            for input, label in self.trainloader:
                input_size = input.shape
                output_size = label.shape
                break

            trace = SkippableTracing(self.args.gpu, self.model, input_size, output_size)
            torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
            self.model = trace.get_modules()
            self.model = Pipe(self.model, chunks=self.args.chunk)

        elif self.args.framework == "API torch":
            self.model.to(device)

    def setup_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                      shuffle=True, num_workers=2)

    def setup_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def setup_loss_fn(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def run(self):
        raise NotImplementedError("Subclasses must implement the 'run' method.")

class MemoryAllocMode(BenchmarkMode):
    def run(self):
        for epoch in range(self.args.epochs):
            start  = [0] * self.args.gpu
            peaked = [0] * self.args.gpu

            for gpu in range(self.args.gpu):
                start[gpu] = torch.cuda.memory_allocated(gpu)

            if self.args.framework == "Pipeline":
                utils.training_pipeline(self.model, self.trainloader, self.args.gpu, self.optimizer, self.loss_fn)

            elif self.args.framework == "API torch":    
                utils.training_normal(self.model, self.trainloader, device, self.optimizer, self.loss_fn)
            
            for gpu in range(self.args.gpu):
                peaked[gpu] = (torch.cuda.max_memory_allocated(gpu) - start[gpu]) // (2 * 1024)

            print(peaked, end=";")

class ExecTimeMode(BenchmarkMode):
    def run(self):
        for epoch in range(self.args.epochs):
            start_time = time.time()

            if self.args.framework == "Pipeline":
                utils.training_pipeline(self.model, self.trainloader, self.args.gpu, self.optimizer, self.loss_fn)

            elif self.args.framework == "API torch":    
                utils.training_normal(self.model, self.trainloader, device, self.optimizer, self.loss_fn)

            end_time = time.time()

            execution_time = end_time - start_time

            print(execution_time, end=";")

class AccuracyMode(BenchmarkMode):
    def run(self):
        # Add accuracy measurement logic here
        pass

def main():
    parser = argparse.ArgumentParser(description="Script d'analyse avec différentes options")

    parser.add_argument(
        "mode",
        choices=["memory_alloc", "exec_time", "accuracy"],
        help="Mode d'analyse (memory_alloc, exec_time, accuracy)"
    )
    parser.add_argument(
        "model",
        choices=["CNN", "Resnet18", "Resnet50", "orbit5k"],
        help="Modèle à utiliser (CNN, Resnet18, Resnet50, orbit5k)"
    )
    parser.add_argument(
        "framework",
        choices=["Pipeline", "API torch"],
        help="Choix du framework (Pipeline, API torch)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="Nombre de GPU à utiliser (par défaut 1)"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=10,
        help="Nombre de chunks (par défaut 10)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre d'époques (par défaut 100)"
    )

    args = parser.parse_args()

    if args.mode == "memory_alloc":
        mode = MemoryAllocMode(args)
    elif args.mode == "exec_time":
        mode = ExecTimeMode(args)
    elif args.mode == "accuracy":
        mode = AccuracyMode(args)
    else:
        raise ValueError("Invalid mode specified.")

    mode.run()

if __name__ == "__main__":
    main()

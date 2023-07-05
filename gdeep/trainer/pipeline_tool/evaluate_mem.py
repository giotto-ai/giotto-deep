from gpu_alloc import TraceMalloc
from dataset import PipelineDataset
from pipelinecache.layered_model import PipelinedModel
import os

import torch
import argparse
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'

parser = argparse.ArgumentParser()
parser.add_argument('--input_shape', type=str, help='Input shape as a list')
parser.add_argument('--output_shape', type=str, help='Output shape as a list')
parser.add_argument('--number_gpu', type=int, help='Number of GPU')
parser.add_argument('--number_chunks', type=int, help='Number of chunks')
args = parser.parse_args()

input_shape = args.input_shape.replace("[", "").replace("]", "")
input_shape = input_shape.split(",")
input_shape = [int(x.strip()) for x in input_shape]

output_shape  = args.output_shape.replace("[", "").replace("]", "")
output_shape  = output_shape.split(",")
output_shape  = [int(x.strip()) for x in output_shape]

number_gpus   = args.number_gpu
number_chunks = args.number_chunks


trace_gpu_alloc = TraceMalloc(number_gpus)
criterion = torch.nn.CrossEntropyLoss()
torch.cuda.init()
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
with trace_gpu_alloc:

    model = PipelinedModel()

    dataset = PipelineDataset(1024, input_shape[1:], [1] if len(output_shape) == 1 else output_shape[1:])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=input_shape[0], shuffle=True)

    model = model.get_modules()
    model = torch.distributed.pipeline.sync.Pipe(model, number_chunks)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for _ in range(3):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(0)
            labels = labels.to(number_gpus- 1)
            
            outputs = model(inputs).local_value()

            # Forward pass
            loss = criterion(outputs, labels.squeeze())

            # Backward pass et mise à jour des poids
            loss.backward()
print(trace_gpu_alloc.peaked)



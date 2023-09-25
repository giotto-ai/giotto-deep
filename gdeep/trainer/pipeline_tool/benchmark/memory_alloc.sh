#!/bin/bash

nb_gpus="$1"

if [ -z "$nb_gpus" ]; then
    echo "Please give the maximum number of GPU you want to use"
    exit 1
fi

rm -f memory_alloc.txt

echo "Framework;Model;Nb GPU;Nb Chunks;Memory Peaks" >> memory_alloc.txt

output=$(python3 benchmark.py memory_alloc CNN "API torch" --gpu 1 --chunk 2 --epoch 2)
echo "API Torch;CNN;1;0;$output" >> memory_alloc.txt

for ((i = 1; i <= nb_gpus; i++)); do
    output=$(python3 benchmark.py memory_alloc CNN "Pipeline" --gpu $i --chunk 2 --epoch 2)
    echo "Pipeline;CNN;$i;2;$output" >> memory_alloc.txt

    output=$(python3 benchmark.py memory_alloc CNN "Pipeline" --gpu $i --chunk 4 --epoch 2)
    echo "Pipeline;CNN;$i;4;$output" >> memory_alloc.txt

    output=$(python3 benchmark.py memory_alloc CNN "Pipeline" --gpu $i --chunk 8 --epoch 2)
    echo "Pipeline;CNN;$i;8;$output" >> memory_alloc.txt
done
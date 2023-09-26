#!/bin/bash

nb_gpus="$1"
nb_epochs="$2"

if [ -z "$nb_gpus" ]; then
    echo "Please provide the maximum number of GPUs you want to use."
    exit 1
fi

rm -f exec_time.txt

time_sequence=$(seq -s ";" -f "Time %g" $nb_epochs)

echo "Framework;Model;Number of GPUs;Number of Chunks;$time_sequence" >> exec_time.txt

# Function for loading effect
function loading {
    local text="$1"
    local pid=$2
    local delay=0.2
    local spin='-\|/'

    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spin#?}
        printf " $text... [%c]  \r" "$spin"
        local spin=$temp${spin%"$temp"}
        sleep $delay
    done
    printf " $text... \e[32m[OK]\e[0m  \n" # The [OK] text is in green
}

# Launch message
echo "---------------------------"
echo "EXECUTION TIME BENCHMARKING"
echo "---------------------------"

echo "Framwork API Torch"
output=$(python3 benchmark.py exec_time CNN "API torch" --gpu 1 --chunk 2 --epoch $nb_epochs)  #&
# loading "   Benchmarking execution time [Model CNN]" $!
echo "API Torch;CNN;1;0;$output" >> exec_time.txt


echo "Framework Pipeline tool"
for ((i = 1; i <= nb_gpus; i++)); do
    # Launch message
    output=$(python3 benchmark.py exec_time CNN "Pipeline" --gpu $i --chunk 2 --epoch $nb_epochs) #&
    #loading "   Benchmarking execution time with $i GPU(s) and 2 chunks [Model CNN]" $!
    echo "Pipeline;CNN;$i;2;$output" >> exec_time.txt

    output=$(python3 benchmark.py exec_time CNN "Pipeline" --gpu $i --chunk 4 --epoch $nb_epochs) #&
    #loading "   Benchmarking execution time with $i GPU(s) and 4 chunks [Model CNN]" $!
    echo "Pipeline;CNN;$i;4;$output" >> exec_time.txt

    output=$(python3 benchmark.py exec_time CNN "Pipeline" --gpu $i --chunk 8 --epoch $nb_epochs) #&
    #loading "   Benchmarking execution time with $i GPU(s) and 8 chunks [Model CNN]" $!
    echo "Pipeline;CNN;$i;8;$output" >> exec_time.txt
done

# Confirmation message
echo "All benchmarks are completed. Results are stored in exec_time.txt."

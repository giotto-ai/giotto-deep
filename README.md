![image](https://github.com/giotto-ai/giotto-deep/blob/master/docs/giotto-deep-big.svg)

![Python package](https://github.com/giotto-ai/giotto-deep/workflows/Python%20package/badge.svg)
![Deploy to gh-pages](https://github.com/giotto-ai/giotto-deep/workflows/Deploy%20to%20gh-pages/badge.svg)
![Upload Python Package](https://github.com/giotto-ai/giotto-deep/workflows/Upload%20Python%20Package/badge.svg)
# giotto-deep

The first library to bring seamless integration between topological data
analysis and deep learning on top of PyTorch.
The code for Persformer will be released open-source soon together
with Giotto-deep.
It aims to make the day-to-day of researchers easy, allowing them
to focus on inventing new models and layers rather than dealing
with the more standard deep-learning code.
It comes with optimized implementations for multi-GPU/TPU
computations and the ability to run benchmarks and
hyperparameter optimization in a few lines of code.

## Documentation

You can find the documentation of this repository here: https://giotto-ai.github.io/giotto-deep/

## Run tensorboard for visualisation

In order to analyse the results of your models, you need to start **tensorboard**. On the terminal, move inside the `/examples` folder. There, run the following command:
```
tensorboard --logdir=runs
```
Afterwards go [here](http://localhost:6006/) and, after running the notebooks of interest, you will see all the visualisation results that you stored in the `writer = SummaryWriter()`.

## Install dev version

The first step to install the developer version of the package is to `git clone` this repository:
```
git clone https://github.com/giotto-ai/giotto-deep.git
```
The change the current working directory to the Repository root folder, e.g. `cd giotto-deep`. 
Once you are in the root folder, install the package dynamically with:
```
pip install -e .
```
Make sure you have upgraded to the last version of `pip` with
```
python -m pip install --upgrade pip
```

## Run local tests
To run both unit and integration tests on *macOS* or *Linux*, simply run the following command from the root folder:
```
bash local_test.bh
```

## TPU support in Google Colab

I order to run your analysis on TPU cores, you ca use the following lines:
```
!git clone https://username:token@github.com/giotto-ai/giotto-deep
!ls
!pip uninstall -y tensorflow
!pip install -e giotto-deep/
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
```
Once you have run the lines above, please make sure to restart the runtime.

The code will automatically detect the TPU core and use it as deffault to run the experiments. GPUs are also automatically supported.

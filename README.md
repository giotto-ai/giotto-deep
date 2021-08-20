![image](https://github.com/giotto-ai/giotto-deep/blob/master/docs/giotto-deep-big.svg)

![Python package](https://github.com/giotto-ai/giotto-deep/workflows/Python%20package/badge.svg)
![Deploy to gh-pages](https://github.com/giotto-ai/giotto-deep/workflows/Deploy%20to%20gh-pages/badge.svg)
![Upload Python Package](https://github.com/giotto-ai/giotto-deep/workflows/Upload%20Python%20Package/badge.svg)
# giotto-deep

WIP
## Documentation

You can find the documentation of this repository here: https://giotto-ai.github.io/giotto-deep/

## Run tensorboard for visualisation

In order to analyse the results of your models, you need to start **tensorboard**. On the terminal, move inside the `/example` folder. There, run the following command:
```
tensorboard --logdir=runs
```
Afterwards go [here](http://localhost:6006/) after running the notebook to see all the visualisation results that you stored in the `writer = SummaryWriter()`.

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

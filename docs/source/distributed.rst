#####################
Distributed computing
#####################

.. _distributed:

************
Introduction
************

Long hyper parameter searches might easily take days. Here, we provide a solution to distribute such computations over a Kubernetes cluster. Kubernetes is available on the major cloud platforms.

****************
K8 configuration
****************

In the folder ``/kubernetes/`` you will find a detailed ``readme.md``file with all the instructions and configuration files needed to set-up the cluster. It is mostly a matter of running a couple of very standard command and the cluster will be up and running.

*********************************
Starting distributed computations
*********************************

The distribution of computations is done automatically, you just to wrap all your HPO details inside a single file. In particular, in the ``/kubernetes/examples/`` folder you will find a notebook and a ``.py`` file. The ``.py`` file contains all the HPO relevant code, which you will have to customise following your needs. The notebook simply launches the enqueueing (and hence the jobs will run on the workers pods).

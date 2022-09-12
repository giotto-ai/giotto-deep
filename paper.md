---
title: 'giotto-deep: A Python Package for Topological Deep Learning'
tags:
  - Python
  - topological data analysis
  - deep learning
  - persformer
authors:
  - name: Matteo Caorsi^[Co-first author, Corresponding author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0001-9416-9090
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Raphael Reinauer^[Co-first author] # note this makes a footnote saying 'Co-first author'
    affiliation: 2
  - name: Nicolas Berkouk
    affiliation: 2
affiliations:
 - name: L2F SA, Rue du Centre 9, Saint-Sulpice, 1025, CH
   index: 1
 - name: EPFL, Lausanne, VAUD (CH)
   index: 2
date: 01 July 2022
bibliography: paper.bib

---

# Summary

Topological data analysis (TDA) has already provided many novel insights to machine learning 
[@carriere2020perslay] due to its capabilities of synthesizing the shape information 
into a multiset of points in two dimensions: the persistence diagrams [@wasserman2016topological]. 
The hope of many researchers in the field is to give new insights into deep-learning
models by applying TDA techniques to study the models weights, the activations values in
the different layers and their evolution during the training phase [@naitzat2020topology].
Orthogonally to that, TDA techniques have been used as feature engineering tools to extract
novel information from the data, which are then used as standard features in a machine learning
pipeline, with significant success in many fields [@hensel2021survey]. 

# Statement of need

`giotto-deep` is an deep-learning Python package that seamlessly integrate topological
data analysis models and data structures. The library is founded on a PyTorch core
due to the extensive use of such library in the machine-learning community.
The API for `giotto-deep` was designed to provide a class-based and user-friendly 
interface to fast implementations of common machine learning tasks, 
such as data preprocessing, model building,
model training and validation, reporting and logging, as well as image classification, 
Q&A, translation, persistence diagram vectorization (via Persformer [@reinauer2021persformer]), 
but also more advanced tasks such as hyper parameter optimization. 
`giotto-deep` also relies heavily on optuna [@akiba2019optuna] for (distributed, multi pod) 
hyper parameter searches.

`giotto-deep` has been designed to be used by both mathematics researchers and by
machine learning engineers. The combination of speed, versatility, design, and 
support for TDA data structures in `giotto-deep` will enable exciting
scientific explorations of the behavior of deep learning models, hopefully shedding 
new lights on the generalisability and robustness of such complex and powerful
models.

![Architecture UML diagram.\label{fig:arch}](gdeep_arch.png)

`giotto-deep` architecture is schematized, using UML, in picture \autoref{fig:arch}.

The hyper parameter searches (HPO) can also be distributed on a `kubernetes` cluster
using [python RQ](https://python-rq.org), speeding up the computation: this is an 
essential aspect when deal with large models and large hyper parameters searches. 
Many topological computations in `giotto-deep` 
are performed by `giotto-tda` [@tauzin2021giotto].

Gitto-deep handles the whole pipeline: from data preprocessing up to the hyper
parameter search, the `k-fold` cross validation and the deployment of the models. 
We provide various preprocessing and training pipelines already implemented, but 
we invite users to extend and improve them. The eventual goal is to create an 
readable code base which is easy to learn and simple to implement, so that the 
contribution of new features is encouraged.
Additionally we provide classical TDA datasets in a specific dataset cloud
where all the classic TDA datasets will be stored, the user can access and download the dataset 
from the cloud in a fully automated way from the `giotto-deep`.

# Research projects using `giotto-deep`

The current most relevant scientific application of this software is the Persformer: 
a novel algorithm to automatize the persistence diagrams vectorization [@reinauer2021persformer].

# Acknowledgements

The authors would like to acknowledge the financial support of the Swiss federation:
Innosuisse project 41665.1 IP-ICT.

# References

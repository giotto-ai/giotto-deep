---
title: 'giotto-deep: A Python package for topological deep learning'
tags:
  - Python
  - topological data analysis
  - deep learning
  - persformer
authors:
  - name: Matteo Caorsi^[Co-first author, Corresponding author] # note this makes a footnote saying 'Co-first author'
    orcid: 0000-0000-0000-0000
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
[@carriere2020perslay] due to its capabilities of synthesising the shape information 
into a multiset of points in two dimensions: the persistence diagrams [@wasserman2016topological]. 
The hope of many researchers in the field is to give new insights into deep-learning
models by applying TDA techniques to study the models weights, the activations values in
the different layers and their evolution during the training phase [@naitzat2020topology]. 
Despite the continuous progress and improvements in the field, the computational 
tools available are scattered all over the Internet and there is currently
little compatibility between the different tools and, in general, between the 
data structures needed in TDA and standard deep-learning frameworks
like PyTorch [@paszke2019pytorch].

# Statement of need

`giotto-deep` is an deep-learning Python package that seamlessly integrate topological
data analysis models and data structures. The library is founded on a PyTorch core
due to the extensive use of such library in the machine-learning community.
The API for `giotto-deep` was designed to provide a class-based and user-friendly 
interface to fast implementations of common machine learning tasks, 
such as data preprocessing, model building,
model training and validation, reporting and logging, as well as image classification, 
Q&A, translation, persistence diagram vectorisation (via Persformer [@reinauer2021persformer]), 
but also more advanced tasks such as hyper parameter optimisation. 
`giotto-deep` also relies heavily on optuna [@akiba2019optuna] for (distributed, multi pod) 
hyper parameter searches.

`giotto-deep` has been designed to be used by both mathematics researchers and by
machine learning engineers. The combination of speed, versatility, design, and 
support for TDA data structures in `giotto-deep` will enable exciting
scientific explorations of the behavior of deep learning models, hopefully sheding 
new lights on the generalisability and robustness of such complex and powerful
models.

![Architecture UML diagram.\label{fig:arch}](gdeep_arch.png)

`giotto-deep` architecture is schematised, using UML, in picture \autoref{fig:arch}.


The hyper perameter searches (HPO) can aslo be distributed on a `kubernetes` cluster
using [python RQ](https://python-rq.org). Many topological computations in `giotto-deep` 
are performed by `giotto-tda` [@tauzin2021giotto].

# Mentions

The current most relevant application of this software is the Persformer: a novel
algorithms to automatically transform persistence diagrams into vectors [@reinauer2021persformer].

# Acknowledgements

The authors acknowledge the financial support of the nnosuisse project 41665.1 IP-ICT.

# References

############
Introduction
############

.. _introduction:

******
Goal
******

The goal of ``giotto-deep`` is to bring topological techniques to deep-learning. Furthermore, the high-level API of the library allows quickly to load data and train models: researchers can thus
focus on creating new layers, models, loss functions, metrics,... and leave the more dull steps to
``giotto-deep``.

************************
Short technical overview
************************

In ``giotto-deep`` there are a few objects that require your attention. Each of the following
sections will explain the most important ones.

DatasetBuilder and DataLoaderBuilder
====================================

You can easily create or load datasets using the ``DatasetBuilder`` class, apply preprocessing and 
then get them ready to train using the ``DataLoaderBuilder`` class.

.. code-block:: python

   bd = DatasetBuilder(name="AG_NEWS", convert_to_map_dataset=True)
   ds_tr_str, ds_val_str, ds_ts_str = bd.build()
   ptd = TokenizerTextClassification()
   ptd.fit_to_dataset(ds_tr_str)
   transformed_textds = ptd.attach_transform_to_dataset(ds_tr_str)
   transformed_textts = ptd.attach_transform_to_dataset(ds_val_str)
   dl_tr2, dl_ts2, _ = DataLoaderBuilder((transformed_textds, transformed_textts)).build()


Model
======

Models are vanilla ``torch.nn.Module`` and subclasses of it. You have all the freedom of PyTorch to 
Build your custom model.

Trainer
=======

Once you have setup your data, your model, your loss function (and maybe a performance metric), you
can put all these ingredients together into the  ``Trainer`` class. Then, with the ``train`` method
you can run the training and validation of your models, specifying all the parameters the you need. 

.. code-block:: python

   writer = SummaryWriter()
   loss_fn = nn.CrossEntropyLoss()
   pipe = Trainer(model, (dl_tr2, dl_ts2), loss_fn, writer)
   pipe.train(SGD, 7, False, {"lr": 0.01}, {"batch_size": 20})


HyperPatameterOptimisation
==========================

Instead of a single training, you may also want to search the space of hyper parameters to find the 
Best model possible. In ``giotto-deep`` this step can be done in a few lines using the class 
``HyperPatameterOptimisation``:

.. code-block:: python

   search = HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)
   optimizers_params = {"lr": [0.001, 0.01]}
   dataloaders_params = {"batch_size": [32, 64, 16]}
   models_hyperparams = {"n_nodes": ["200"]}
   search.start(
       (SGD, Adam),
       3,
       False,
       optimizers_params,
       dataloaders_params,
       models_hyperparams,
       n_accumulated_grads=2
   )


Interpreter
===========

Once you have trained your model you can both check the losses and metric on ``tensorboard`` or use
more advanced interpretability tools on the model performances. The interpretability tools are part
of the ``Interpreter`` class.

.. code-block:: python

   inter = Interpreter(pipe.model, method="LayerIntegratedGradients")
   inter.interpret_text("I am writing about money and business", 
       0, 
       ptd.vocabulary,
       ptd.tokenizer,
       layer=pipe.model.embedding,
       n_steps=500,
       return_convergence_delta=True
   )


Visualizer
==========

You can visualise additional results, like persistence diagrams of the activations, the model graph
or the heat maps resulting form the `Interpreter``: you need the ``Visualiser`` class.

.. code-block:: python

   vs = Visualiser(pipe)
   vs.plot_interpreter_text(inter)
   vs.plot_data_model()


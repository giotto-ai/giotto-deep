.. _mixed precision: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
.. _backward prefetch: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch
.. _FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp-works
.. _Advanced FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html

.. _parallel:
#####################
Parallel training
#####################

************
Introduction
************

When multiple GPUs are available, it is possible to use more than one to train the same model. Currently, two methods of multi-GPU training are available in giotto-deep and provide different benefits: Data parallelism and Pipeline parallelism

**********************
Data parallelism: FSDP
**********************

Data parallelism consists in training multiple copies of the model on partitions of the dataset. For example: one GPU may be responsible for training one copy of the model on one half of the training dataset while another trains another copy on the second half of the dataset. At the end or multiple times during a training epoch, both copies are merged to obtain a new model technically trained on the whole dataset. This method relies on the assumption that the merging of the half-trained models is fast enough and results in good enough improvements to compensate the half-training. The principle is the same for more GPUs but the performance may not improve significantly past a certain point. in giotto-deep, this method is called DDP

A more complex method consists in sharding the model to distribute it across multiple GPUs. During the training, one GPU can request the shard it needs from the GPU responsible for this shard, do its calculations and then discard the shard. Once the gradients for this shard is calculated in the backward pass, the shard is retrieved again, calculations are done and the updated shard is sent back to the responsible GPU (as explained in the `FSDP tutorial`_). More communication between the GPUs is required as a result but this approach saves memory. This algorithm is called ZERO and exists in 2 variants in giotto-deep: ZERO2 doesn't discard the shard after the forward pass and thus saves time in communication but requires more memory, ZERO3 discards the shard and thus requires more communication but less memory. In itself, using one of those 2 algorithms may not result in better training time but the freed memory can be used to increase the batch size (which usually results in faster epochs) or increase the size of the model

Some optimisations allow those algorithms to increase in performance even more:
* Mixed precision: Converts the weights, gradients or the transmitted data to a lower precision to speed up calculation (potentially using the hardware support of the GPU) or transmission of data between the GPUs
* Backward prefetch: Allows shards to be recovered in different ways to optimise memory usage or performance

More optimisations are discussed in the `Advanced FSDP tutorial`_ 

The Data parallelism algorithm and optimisations to use are very dependant on the model and training method but can sensibly improve the training time when chosen and configured correctly. Please note however that these methods are currently only available if the model fits entirely in one GPU's memory

Those algorithms are implemented into giotto-deep using pytorch's FSDP tool The implementation's architecture is explained in the diagram below. 

.. image:: _img/giotto_trainer_fsdp.png

To use one of those algorithms, import and instantiate the `Parallelism` class with the following informations:

* p_type: The algorithm to use, defined in the `ParallelismType` enum
    * `DDP`
    * `FSDP_ZERO2`
    * `FSDP_ZERO3`
* devices: List of the available GPUs available on the machine. The list can be generated using `list(range(torch.cuda.device_count()))`. If no list is provided, the class will look for the devices itself
* nb_device: The actual number of GPUs from the devices list to use for the training. Not providing this parameter or providing a value smaller than 0 results in all the devices (found or provided) to be used. 0 or values higher than the number of devices (found or provided) will result in an error
* fetch_type: The shard recovery strategy to use. See Pytorch's documentation on `backward prefetch`_ for more details
* transformer_layer_class: Pytorch module containing the Multi-Head Attention and Feed Forward layers of the used transformer (if used model is a transformer). See the `Advanced FSDP tutorial`_ for more details

The instance can then be given to the `parallel` argument of the `train` function

.. code-block::
    # FSDP not used for training
    valloss, valacc = train.train(SGD, 
                                  args.n_epochs, 
                                  args.cv, 
                                  {"lr": 0.001, "momentum": 0.9})

    # FSDP configured and used for training
    devices = list(range(torch.cuda.device_count()))

    fetch = BackwardPrefetch.BACKWARD_PRE
    
    parallelism = Parallelism(ParallelismType.FSDP_ZERO2,
                                devices, 
                                len(devices),
                                fetch_type=fetch)

    valloss, valacc = train.train(SGD, 
                                  args.n_epochs, 
                                  args.cv, 
                                  {"lr": 0.001, "momentum": 0.9},
                                  parellel=parallelism)

FSDP in giotto-deep works with profiling and cross-validation

.. warning::
    As FSDP uses multiprocessing, it is necessary to use the idiom `if __name__ == __main__:` for the main code. This also implies that the model and datasets should be serialisable (which is not the case of 'to_map_style_dataset' datasets for example)

.. note::
    When using FSDP, the batch size given to the dataloader is used by each GPU. For example, using a batch size of 4 with 2 GPUs effectively corresponds to using a batch size of 8 without FSDP

***********************************
Pipeline parallelism: pipeline-tool
***********************************
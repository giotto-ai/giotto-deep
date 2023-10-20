.. _FSDP documentation: https://pytorch.org/docs/stable/fsdp.html
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

A more complex method consists in sharding the model to distribute it across multiple GPUs. During the training, one GPU can request the shard it needs from the GPU responsible for this shard, do its calculations and then discard the shard. Once the gradients for this shard is calculated in the backward pass, the shard is retrieved again, calculations are done and the updated shard is sent back to the responsible GPU (as explained in the `FSDP tutorial`_). More communication between the GPUs is required as a result but this approach results in lower peak memory use. This algorithm is called ZERO and exists in 2 variants in FSDP: ZERO2 (SHARD_GRAD_OP) doesn't discard the shard after the forward pass and thus saves time in communication but requires more memory, ZERO3 (FULL_SHARD) discards the shard everytime after it is done with its calculations and thus requires more communication but less memory. In itself, using one of those 2 algorithms may not result in better training time but the freed memory can be used to increase the batch size (which usually results in faster epochs) or increase the size of the model

Some optimisations allow those algorithms to increase in performance even more:
* Mixed precision: Converts the weights, gradients or the transmitted data to a lower precision to speed up calculation (potentially using the hardware support of the GPU) or transmission of data between the GPUs
* Backward prefetch: Allows shards to be recovered in different ways to optimise memory usage or performance
* ...

More optimisations are discussed in the `Advanced FSDP tutorial`_ 

The Data parallelism algorithms and optimisations to use are very dependant on the model and training method but can sensibly improve the training time when chosen and configured correctly. 

Those algorithms are implemented into giotto-deep using pytorch's FSDP tool The implementation's architecture is explained in the diagram below. 

.. image:: _img/giotto_trainer_fsdp.png

What the diagram shows is that for each device used for the training, giotto-deep's Trainer will create a new process that executes a subinstance of Trainer. The members of the base instance (model, dataloaders,...) are deepcopied into the subinstances so each subinstance may works on its members without affecting the other processes. The training occurs on each process on its dedicated device. Once the training is complete, the model is retrieved by the subprocess with rank 0 and stored in a temporary file where it is then recovered by the base instance to update the model.

This architecture was selected because it required the least amount of changes to the pre-existing Trainer class of giotto-deep. However, it comes with a few limitations. The examples of "native" FSDP (FSDP not used in a library) show the model and datasets/dataloaders being instanced in each subprocess. However, due to giotto-deep's API, this wasn't possible as the Trainer class expects to be given instances of the model and dataloaders. In order to use FSDP within giotto-deep without some major changes to the API and pre-existing implementation, we had to deepcopy and send the model and dataloaders to each subinstance of Trainer. This means that the dataloaders and models HAVE TO be serialisable using pytorch's pickler. Currently, some features of Pytorch (ex:Map_style_dataset) and giotto-deep (ex:TransformingDataset) aren't serialisable and thus cannot be used as is when trying to train a model with FSDP through giotto-deep. 

To use one of those algorithms, import and instantiate the `Parallelism` class with the following informations:

* p_type: The algorithm to use, defined in the `ParallelismType` enum
    * `PIPELINE`
    * `FSDP`
* devices: List of the GPUs available on the machine. The list can be generated using `list(range(torch.cuda.device_count()))`. If no list is provided, the class will look for the devices itself
* nb_device: The actual number of GPUs from the devices list to use for the training. Not providing this parameter or providing a value smaller than 1 results in all the devices (found or provided) to be used. Values higher than the number of devices (found or provided) will result in an error
* config_fsdp: Dictionnary containing the arguments for the instantiation of FullyShardedDataParallel as per the official `FSDP documentation`_. This allows the user to configure FSDP as he wishes. The device_id parameter of FSDP is automatically handled

The instance can then be given to the `parallel` argument of the `train` function

.. code-block::
    # FSDP not used for training
    valloss, valacc = train.train(SGD, 
                                  args.n_epochs, 
                                  args.cv, 
                                  {"lr": 0.001, "momentum": 0.9})

    # FSDP configured and used for training
    devices = list(range(torch.cuda.device_count()))

    config_fsdp = {
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
        "auto_wrap_policy": always_wrap_policy,
        }

    
    parallelism = Parallelism(ParallelismType.FSDP,
                                devices, 
                                len(devices),
                                config_fsdp=config_fsdp)

    valloss, valacc = train.train(SGD, 
                                  args.n_epochs, 
                                  args.cv, 
                                  {"lr": 0.001, "momentum": 0.9},
                                  parallel=parallelism)

FSDP in giotto-deep works with profiling and cross-validation but not with parallel TPUs. 

.. warning::
    As FSDP uses multiprocessing, it is necessary to use the idiom `if __name__ == __main__:` for the main code. This also implies that the model and datasets should be serialisable (which is not the case of 'to_map_style_dataset' datasets for example)

.. warning::
    Using FSDP without a wrapper will behave as if FSDP weren't used at all

.. note::
    When using FSDP, the batch size given to the dataloader is used by each GPU. For example, using a batch size of 4 with 2 GPUs effectively corresponds to using a batch size of 8 without FSDP

***********************************
Pipeline parallelism: pipeline-tool
***********************************
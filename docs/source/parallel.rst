.. _mixed precision: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
.. _backward prefetch: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch

.. _parallel:
#####################
Parallel training
#####################

************
Introduction
************

When multiple GPUs are available, it is possible to use more than one to train the same model. Currently, two methods of multi-GPU training are available in Giotto and provide different benefits: Data parallelism and Pipeline parallelism

**********************
Data parallelism: FSDP
**********************

Data parallelism consists in training multiple copies of the model on partitions of the dataset. For example: one GPU may be responsible for training one copy of the model on one half of the training dataset while another trains another copy on the second half of the dataset. At the end or multiple times during a training epoch, both copies are merged to obtain a new model technically trained on the whole dataset. This method relies on the assumption that the merging of the half-trained models is fast enough and results in good enough improvements to compensate the half-training. The principle is the same for more GPUs but the performance may not improve significantly past a certain point. in Giotto, this method is called DDP

A more complex method consists in sharding the model to distribute it across multiple GPUs. During the training, one GPU can request the shard it needs from the GPU responsible for this shard, do its calculations and then discard the shard. Once the gradients for this shard is calculated in the backward pass, the shard is retrieved again, calculations are done and the updated shard is sent back to the responsible GPU. This algorithm is called ZERO and exists in 2 variants in Giotto: ZERO2 doesn't discard the shard after the forward pass and thus saves time in communication but requires more memory, ZERO3 discards the shard and thus requires more communication but less memory. In itself, only using one of those 2 algorithms may not result in better training time but the freed memory can be used to increase the batch size which usually results in faster epochs

Some optimisations allow those algorithms to increase in performance even more:
* Mixed precision: Converts the weights, gradients or the transmitted data to a lower precision to speed up calculation (potentially using the hardware support of the GPU) or transmission of data between the GPUs
* Backward prefetch: Allows shards to be recovered in different ways to optimise memory usage or performance

The Data parallelism algorithm and optimisations to use are very dependant on the model and training method but can sensibly improve the training time when chosen and configured correctly. Please note however that these methods are only available and/or interesting if the model fits entirely in the GPU's memory

Those algorithms are implemented into Giotto using pytorch's FSDP tool. To use one of those algorithms, import and instantiate the `Parallelism` class with the following informations:

* p_type: The algorithm to use, defined in the `ParallelismType` enum
    * `DDP`
    * `FSDP_ZERO2`
    * `FSDP_ZERO3`
* devices: List of the available GPUs available on the machine. The list can be generated using `list(range(torch.cuda.device_count()))`
* world_size: The actual number of GPUs from the devices list to use for the training
* mixed_precision: A `torch.distributed.fsdp.MixedPrecision` object that defines the mixed precision strategy to use (or `None` to disable it). See Pytorch's documentation on `mixed precision`_ for more details. Careful: the "torch.bfloat16" precision doesn't have hardware support on every platform: Trying to use it on an unsupported platform will result in an exception
* fetch_type: The shard recovery strategy to use. See Pytorch's documentation on `backward prefetch`_ for more details

The instance can then be given to the `parallel` argument of the `train` function

***********************************
Pipeline parallelism: pipeline-tool
***********************************
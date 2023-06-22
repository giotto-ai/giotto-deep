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

A complementary, more complex method consists in sharding the model to distribute it across multiple GPUs. During the training, one GPU can request the shard it needs from the GPU responsible for this shard, do its calculations and then discard the shard. Once the gradients for this shard is calculated in the backward pass, the shard is retrieved again, calculations are done and the updated shard is sent back to the responsible GPU. This algorithm is called ZERO and exists in 2 variants in Giotto: ZERO2 doesn't discard the shard after the forward pass and thus saves time in communication but requires more memory, ZERO3 discards the shard and thus requires more communication but less memory. In itself, only using one of those 2 algorithms may not result in better training time but the freed memory can be used to increase the batch size which usually results in faster epochs

The Data parallelism algorithm to use is very model and training method dependant but can sensibly improve the training time

***********************************
Pipeline parallelism: pipeline-tool
***********************************
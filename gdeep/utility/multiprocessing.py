import multiprocessing as pmp

import torch.multiprocessing as tmp
from torch.multiprocessing.spawn import _wrap


# This code is heavily inspired from
# https://github.com/pytorch/pytorch/blob/v1.13.1/torch/multiprocessing/spawn.py#L178
def spawn(fn, args=(), nprocs=1):
    mp = pmp.get_context("spawn")
    error_queues = []
    processes = []
    return_queue = mp.SimpleQueue()
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        wrap_args = (*args, return_queue if i == 0 else None)
        process = mp.Process(
            target=_wrap,
            args=(fn, i, wrap_args, error_queue),
            daemon=False,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = tmp.ProcessContext(processes, error_queues)

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass

    return return_queue.get()

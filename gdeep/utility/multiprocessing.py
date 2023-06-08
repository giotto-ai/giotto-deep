import multiprocessing as pmp
import torch.multiprocessing as tmp
from torch.multiprocessing.spawn import _wrap

# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def spawn(fn, args=(), nprocs=1):
    mp = pmp.get_context('spawn')
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
    

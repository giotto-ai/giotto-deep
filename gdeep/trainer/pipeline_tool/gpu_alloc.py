import torch

def BToMb(x):
    """Convert bytes to megabytes.
    
    :param x: Value in bytes.
    :type x: int
    :return: Value in megabytes.
    :rtype: int
    """
    return (x // (2 * 1024))

class TraceMalloc():
    def __init__(self, nb_gpu):
        """Initialize a TraceMalloc object.
        
        :param nb_gpu: Number of GPUs.
        :type nb_gpu: int
        """
        self.nb_gpu = nb_gpu
        self.begin  = [0] * nb_gpu
        self.end    = [0] * nb_gpu
        self.peak   = [0] * nb_gpu
        self.peaked = [0] * nb_gpu

    def __enter__(self):
        """Enter the context manager. 
        
        Save the current memory allocated to all GPUs.
        
        :return: The TraceMalloc object.
        :rtype: TraceMalloc
        """
        for device in range(self.nb_gpu):
            self.begin[device] = torch.cuda.memory_allocated(device)
            
        return self
    
    def __exit__(self, *exc):
        """Exit the context manager. 

        Get all the memory information, allocated and peak, to calculate the true peak between the enter and exit call.
        """
        for device in range(self.nb_gpu):
            self.end[device]    = torch.cuda.memory_allocated(device)
            self.peak[device]   = torch.cuda.max_memory_allocated(device)
            self.peaked[device] = BToMb(self.peak[device] - self.begin[device])
            torch.cuda.reset_peak_memory_stats(device)

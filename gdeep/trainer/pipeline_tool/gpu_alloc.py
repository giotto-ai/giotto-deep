import torch

def BToMb(x): return (x // (2*1024))

class TraceMalloc():
    def __init__(self, nb_gpu):
        self.nb_gpu = nb_gpu
        self.begin  = [0] * nb_gpu
        self.end    = [0] * nb_gpu
        self.peak   = [0] * nb_gpu
        self.peaked = [0] * nb_gpu

    def __enter__(self):
        for device in range(self.nb_gpu):
            self.begin[device] = torch.cuda.memory_allocated(device)
            
        return self
    
    def __exit__(self, *exc):

        for device in range(self.nb_gpu):
            self.end[device]    = torch.cuda.memory_allocated(device)
            self.peak[device]   = torch.cuda.max_memory_allocated(device)
            self.peaked[device] = BToMb(self.peak[device] - self.begin[device])
            torch.cuda.reset_peak_memory_stats(device)

        # for device in range(self.nb_gpu):
        #     print(f"GPU n°{device}")
        #     print(f"    Memory begin -> {BToMb(self.begin[device])} MB")
        #     print(f"    Memory end   -> {BToMb(self.end[device])} MB")
        #     print(f"    Memory peak  -> {BToMb(self.peak[device])} MB")
        #     print(f"    Memory peaked  -> {self.peaked[device]} MB")
       

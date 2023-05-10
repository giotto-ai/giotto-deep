from torch.utils.data.sampler import Sampler
import torch
from typing import Optional, TypeVar, Iterator, Sequence
import math

T_co = TypeVar('T_co', covariant=True)

class GiottoSampler(Sampler[T_co]):
    def __init__(self, 
                indices: Sequence[T_co],
                shuffle: bool = True,
                world_size: int = 1,
                rank: int = 0,
                ) -> None:
        self.indices = indices
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.indices_per_rank = math.ceil(len(self.indices) / self.world_size)
        self.total_size =  self.indices_per_rank * self.world_size
        self.padding_size = self.total_size - len(self.indices)
        self.epoch = 0

    def __iter__(self) -> Iterator[T_co]:
        # Shuffle if needed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            curr_indices_perm = torch.randperm(len(self.indices), generator=g).tolist()
        else:
            curr_indices_perm = list(self.indices)

        # Add padding to indices list if needed
        if self.padding_size <= len(self.indices):
            curr_indices_perm += curr_indices_perm[:self.padding_size]
        else:
            curr_indices_perm += (curr_indices_perm * math.ceil(self.padding_size / len(curr_indices_perm)))[:self.padding_size]

        # Subsample indices for current rank
        indices = curr_indices_perm[self.rank:self.total_size:self.world_size]

        for i in indices:
            yield self.indices[i]

    def __len__(self) -> int:
        return self.indices_per_rank

    def new_epoch(self):
        self.epoch += 1
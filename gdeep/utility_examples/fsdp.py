import enum
from torch.distributed.fsdp import ShardingStrategy

class ShardingStrategyEx(enum.Enum):
    FULL_SHARD = enum.auto()
    SHARD_GRAD_OP = enum.auto()
    NO_SHARD = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_str(s: str):
        try:
            return ShardingStrategyEx[s]
        except KeyError:
            raise ValueError(f"Unknown {s}")

    def to_ss(self) -> ShardingStrategy:
        if self is ShardingStrategyEx.FULL_SHARD:
            return ShardingStrategy.FULL_SHARD
        elif self is ShardingStrategyEx.SHARD_GRAD_OP:
            return ShardingStrategy.SHARD_GRAD_OP
        elif self is ShardingStrategyEx.NO_SHARD:
            return ShardingStrategy.NO_SHARD
        else:
            raise ValueError(f"Unknown {self}")

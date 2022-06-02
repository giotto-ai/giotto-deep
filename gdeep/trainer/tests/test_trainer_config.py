
from gdeep.trainer import TrainerConfig
from torch.optim import SGD


def test_trainer_config() -> None:
    config = TrainerConfig(SGD)
    dictionary = config.to_dict()
    assert isinstance(dictionary, dict)


from gdeep.search import HPOConfig
from torch.optim import SGD


def test_hpo_config() -> None:
    config = HPOConfig([SGD])
    dictionary = config.to_dict()
    assert isinstance(dictionary, dict)

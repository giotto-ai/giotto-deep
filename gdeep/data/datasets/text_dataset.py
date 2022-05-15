import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union
from ..transforming_dataset import TransformingDataset

from gdeep.utility import DEVICE

Tensor = torch.Tensor


class TextDataset(TransformingDataset):
    """This class is the base class for the text-datasets.
    The source dataset for this class are expected to be
    dataset of the form ``(label, string)``.

    Args:
        dataset (torch Dataset):
            The source dataset for this class.
            It is expected to be a
            dataset of the form ``(label, string)``.
        transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``


    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import TextDataset, TokenizerTextClassification

        dl = TorchDataLoader(name="AG_NEWS", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDataset(dl_tr.dataset,
                             TokenizerTextClassification())

    """
    pass


class TextDatasetTranslation(TextDataset):
    """This class is the class for the text datasets
    dealing with translation tasks. The source data is expected
    to be of the form ``(string, string)`` containing the
    senteces to translate (left string translates into right string).

    Args:
        dataset (torch Dataset):
            The source dataset for this class.
            It is expected to be a
            dataset of the form ``(label, string)``.
        transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``
        target_transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``

    Returns:
        tensor, tensor:
            the first tensor has three dimensions, ``(BS, 2, MAX_LENGTH)``;
            this tensor represents the stacking of both the tokenisation
            of the source and target sentence.

    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import TextDatasetTranslation, PreprocessTextTranslation

        dl = TorchDataLoader(name="Multi30k", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDatasetTranslation(dl_tr.dataset,
            PreprocessTextTranslation(), None)

    """
    pass


class TextDatasetQA(TextDataset):
    """This class is the class for the text datasets
    dealing with Q&A tasks. The source data is expected
    to be of the form ``(string, string, list[string], list[int])`` containing the
    senteces ``(context, question, [answer(s)],
    [initial position (in characters) of the answer(s)]))``.

    Args:
        dataset (torch Dataset):
            The source dataset for this class.
            It is expected to be a
            dataset of the form ``(label, string)``.
        transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``
        target_transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``

    Returns:
        tensor, tensor:
            the first tensor has three dimensions, ``(BS, 2, MAX_LENGTH)``;
            this tensor represents the stacking of both the tokenisation
            of the context and question sentence.
            The second tensor is just the batch of pairs of start and
            end positions of the answers

    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import  TextDatasetQA, PreprocessTextQA, PreprocessTextQATarget

        dl = TorchDataLoader(name="SQuAD2", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDatasetQA(dl_tr_str.dataset,
                               PreprocessTextQA(),
                               PreprocessTextQATarget())

    """
    pass


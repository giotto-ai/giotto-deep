import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Callable, Generic, NewType, Tuple, \
    Union, Any, Optional, List

import jsonpickle
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchvision.transforms import Resize, ToTensor

from gdeep.utility import DEVICE
from ..abstract_preprocessing import AbstractPreprocessing


# type definition
Tensor = torch.Tensor


class TokenizerTextClassification(AbstractPreprocessing[Tuple[Any, str],
                                                 Tuple[Tensor, Tensor]]):
    """Preprocessing class. This class is useful to convert the data format
    ``(label, text)`` into the proper tensor format ``( word_embedding, label)``

    Args:
        tokenizer :
            the tokenizer of the source text
        vocabulary :
            the vocubulary; it can be built of it can be
            given.

    """


    max_length:int
    is_fitted:bool
    vocabulary: Optional[Sequence[str]]
    tokenizer: Optional[Callable[[str], List[str]]]
    counter: Counter[List[str]]

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]]=None,
                 vocabulary: Optional[Sequence[str]]=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer

        self.vocabulary = vocabulary
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, dataset:Dataset[Tuple[Any, str]]) -> None:
        """Method to extract global data, like to length of
        the sentences to be able to pad.

        Args:
            data (iterable):
                the data in the format ``(label, text)``
        """

        counter = Counter()  # for the text
        for (label, text) in dataset:
            if isinstance(text, tuple) or isinstance(text, list):
                text = text[0]
            counter.update(self.tokenizer(text))
            self.max_length = max(self.max_length, len(self.tokenizer(text)))
        # self.vocabulary = Vocab(counter, min_freq=1)
        if not self.vocabulary:
            self.vocabulary = Vocab(counter)
        self.is_fitted = True
        #self.save_pretrained(".")

    def __call__(self, datum: Tuple[Any, str]) -> Tuple[Tensor, Tensor]:
        """This method is applied to each batch and
        transforms it following the rule below

        Args:
            datum (tuple):
                a single datum, being it a tuple
                with ``(label, text)``
        """
        if not self.is_fitted:
            self.load_pretrained(".")
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        pad_item = self.vocabulary["."]

        _text = datum
        if isinstance(_text, tuple) or isinstance(_text, list):
            _text = _text[1]
        processed_text = torch.tensor(text_pipeline(_text),
                                      dtype=torch.long).to(DEVICE)
        # convert to tensors (padded)
        out_text = torch.cat([processed_text,
                   pad_item * torch.ones(self.max_length - processed_text.shape[0]
                                              ).to(DEVICE)]).to(torch.long)
        # preprocess labels
        label_pipeline = lambda x: torch.tensor(x, dtype=torch.long) - 1

        _label = datum[0]
        try:
            out_label = label_pipeline(_label).to(DEVICE)
        except TypeError:
            if isinstance(_label, tuple) or isinstance(_label, list):
                _label = _label[0]
                out_label = label_pipeline(_label).to(DEVICE)
        return out_text, out_label


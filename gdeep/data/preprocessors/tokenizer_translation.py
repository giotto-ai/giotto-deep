import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, Generic, NewType, Tuple, Union

import jsonpickle
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchvision.transforms import Resize, ToTensor
from gdeep.utility import DEVICE

from ..abstract_preprocessing import AbstractPreprocessing
from ..transforming_dataset import TransformingDataset

# type definition
Tensor = torch.Tensor



class TokenizerTranslation(AbstractPreprocessing):
    """Class to preprocess text dataloaders for translation
    tasks. The Dataset type is supposed to be ``(string, string)``

        Args:
            vocabulary (torch Vocabulary):
                the vocubulary of the source text;
                it can be built automatically or it can be
                given.
            vocabulary_target (torch Vocabulary):
                the vocubulary of the target text;
                it can be built automatically or it can be
                given.
            tokenizer (torch Tokenizer):
                the tokenizer of the source text
            tokenizer_target (torch Tokenizer):
                the tokenizer of the target text

    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import TokenizerTranslation, TextDatasetTranslation

        dl = TorchDataLoader(name="Multi30k", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDatasetTranslation(dl_tr.dataset,
            TokenizerTranslation())

        """

    def __init__(self, vocabulary=None,
                 vocabulary_target=None,
                 tokenizer=None,
                 tokenizer_target=None):

        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        if tokenizer_target is None:
            self.tokenizer_target = get_tokenizer('basic_english')
        else:
            self.tokenizer_target = tokenizer_target
        self.vocabulary = vocabulary
        self.vocabulary_target = vocabulary_target
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, data):
        counter = Counter()  # for the text
        counter_target = Counter()  # for the text
        for text in data:
            counter.update(self.tokenizer(text[0]))
            counter_target.update(self.tokenizer_target(text[1]))
            self.max_length = max(self.max_length, len(self.tokenizer(text[0])))
            self.max_length = max(self.max_length, len(self.tokenizer_target(text[1])))
        # self.vocabulary = Vocab(counter, min_freq=1)
        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        if self.vocabulary_target is None:
            self.vocabulary_target = Vocab(counter_target)
        self.is_fitted = True
        #self.save_pretrained(".")

    def __call__(self, datum):
        """This method is applied to each batch and
        transforms it following the rule below

        Args:
            datum (torch.tensor):
                a single datum
        """

        if not self.is_fitted:
            self.load_pretrained(".")
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]
        text_pipeline_target = lambda x: [self.vocabulary_target[token] for token in
                                   self.tokenizer_target(x)]

        pad_item = self.vocabulary["."]
        pad_item_target = self.vocabulary_target["."]

        processed_text = torch.tensor(text_pipeline(datum[0]),
                                      dtype=torch.long).to(DEVICE)
        processed_text_target = torch.tensor(text_pipeline_target(datum[1]),
                                      dtype=torch.long).to(DEVICE)
        # convert to tensors (padded)
        out_source = torch.cat([processed_text,
                                pad_item * torch.ones(self.max_length - processed_text.shape[0]
                                               ).to(DEVICE)]).to(torch.long)
        out_target = torch.cat([processed_text_target,
                                pad_item_target * torch.ones(self.max_length - processed_text_target.shape[0]
                                               ).to(DEVICE)]).to(torch.long)
        X = torch.stack([out_source.to(torch.long), out_target.to(torch.long)])
        y = out_target.to(torch.long)  #.clone().detach()
        return X, y


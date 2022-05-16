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



class TokenizerQA(AbstractPreprocessing):
    """Class to preprocess text dataloaders for Q&A
    tasks. The type of dataset is assumed to be of the
    form ``(string,string,list[string], list[string])``.

    Args:
        vocabulary (torch Vocab):
            the torch vocabulary
        tokenizer (torch Tokenizer):
            the tokenizer of the source text

    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import  TextDatasetQA, TokenizerQA

        dl = TorchDataLoader(name="SQuAD2", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDatasetQA(dl_tr_str.dataset,
                               TokenizerQA())

    """

    def __init__(self, vocabulary=None,
                 tokenizer=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, data):

        counter = Counter()  # for the text
        for (context, question, answer, init_position) in data:
            if isinstance(context, tuple) or isinstance(context, list):
                context = context[0]
            counter.update(self.tokenizer(context))
            self.max_length = max(self.max_length, len(self.tokenizer(context)))
            if isinstance(question, tuple) or isinstance(question, list):
                question = question[0]
            counter.update(self.tokenizer(question))
            self.max_length = max(self.max_length, len(self.tokenizer(question)))

        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        self.pad_item = self.vocabulary["."]
        self.is_fitted = True
        # self.save_pretrained(".")

    def __call__(self, datum:tuple) -> Tuple[Tensor, Tensor]:
        if not self.is_fitted:
            self.load_pretrained(".")
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        processed_context = torch.tensor(text_pipeline(datum[0]),
                                      dtype=torch.int64).to(DEVICE)
        out_context = torch.cat([processed_context,
                         self.pad_item * torch.ones(self.max_length - processed_context.shape[0]
                                               ).to(DEVICE)])
        processed_question = torch.tensor(text_pipeline(datum[1]),
                                         dtype=torch.int64).to(DEVICE)

        out_question = torch.cat([processed_question,
                         self.pad_item * torch.ones(self.max_length - processed_question.shape[0]
                                               ).to(DEVICE)])

        pos_init_char = datum[3][0]
        pos_init = len(self.tokenizer(datum[0][:pos_init_char]))
        pos_end = pos_init + len(self.tokenizer(datum[2][0]))

        return [torch.stack((out_context, out_question)).to(torch.long),
         torch.stack((torch.tensor(pos_init), torch.tensor(pos_end))).to(torch.long)]




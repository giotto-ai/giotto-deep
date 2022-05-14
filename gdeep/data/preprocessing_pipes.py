import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Generic, NewType, Tuple, Union

import jsonpickle
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchvision.transforms import Resize, ToTensor

from gdeep.data.transforming_dataset import TransformingDataset

from .abstract_preprocessing import AbstractPreprocessing

# type definition
Tensor = torch.Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Normalisation(AbstractPreprocessing[Tensor, Tensor]):
    """This class runs the standard normalisation on all the dimensions of
    the tensors of a dataloader. For example, in case of images where each item is of
    shape ``(C, H, W)``, the average will and the standard deviations
    will be tensors of shape ``(C, H, W)``
    """
    is_fitted: bool
    mean: Tensor
    stddev: Tensor

    def __init__(self):
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[Tensor, Any]]) -> None:
        self.mean = _compute_mean_of_dataset(dataset)
        self.stddev = _compute_stddev_of_dataset(dataset, self.mean)
        self.is_fitted = True

    def __call__(self, item: Tensor) -> Tensor:
        if not self.is_fitted:
            raise RuntimeError("The normalisation is not fitted to any dataset."
                               " Please call fit_to_dataset() first.")
        if not torch.all(self.stddev > 0):
            warnings.warn("The standard deviation contains zeros! Adding 1e-7")
            self.stddev = self.stddev + 1e-7
        out = (item - self.mean)/ self.stddev
        return out

def _compute_mean_of_dataset(dataset: Dataset[Tuple[Tensor, Any]]) -> Tensor:
    """Compute the mean of the whole dataset"""
    mean: Tensor = torch.zeros(dataset[0].shape, dtype=torch.float64, device=DEVICE)
    for idx in range(len(dataset)):  # type: ignore
        if idx == 0:
            mean += dataset[idx][0]
        else:
            mean = (mean * idx + dataset[idx][0]) / (idx + 1)
    return mean

def _compute_stddev_of_dataset(dataset: Tuple[Tensor, Any], mean: Tensor) -> Tensor:
    """Compute the stddev of the whole dataset"""
    mean_normalized_dataset = TransformingDataset(dataset, lambda x: (x - mean)**2)
    stddev: Tensor = _compute_mean_of_dataset(mean_normalized_dataset)
    return stddev.sqrt()


class PreprocessingPipeline(AbstractPreprocessing):
    """class to compose preprocessing classes. The
    order is very important, and to make sure that the output of
    one preprocessing is acceptable as input for the
    following preprocessing

    Args:
        list_of_preproc_and_datatypes (list):
            list of tuples in which the the first
            element if the class instance of the
            AbstractPreprocessing and the second element
            is the Dataset class (not the instance!)

    Examples::

        from torch.utils.data import Dataset
        from gdeep.data import PreprocessingPipeline, Normalisation
        from gdeep.data import PreprocessTextData, TextDataset

        PreprocessingPipeline(((PreprocessTextData(), None, TextDataset),
                               (Normalisation(), None, BasicDataset)))

    """
    def __init__(self, list_of_transforms_and_datatypes:list):
        self.list_of_cls = list_of_transforms_and_datatypes
        
    def fit_to_data(self, dataset):
        for (transform, target_transform, data_type_class, *args) in self.list_of_cls:
            dataset = data_type_class(*args, dataset=dataset,
                                      transform=transform,
                                      target_transform=target_transform)


    def __call__(self, datum:Tensor) -> Tensor:
        for (transform, _, _, *_) in self.list_of_cls:
            datum = transform(datum)
        return datum

    def __len__(self) -> int:
        return len(self.list_of_cls)

    def __getitem__(self, index: int):
        return self.list_of_cls[index]

    def __iter__(self):
        return iter(self.list_of_cls)

    def __repr__(self) -> str:
        return f'PreprocessingPipeline({self.list_of_cls})'

    def __add__(self, other):
        return PreprocessingPipeline(self.list_of_cls + other.list_of_cls)


class PreprocessTextData(AbstractPreprocessing):
    """Preprocessing class. This class is useful to convert the data format
    ``(label, text)`` into the proper tensor format ``( word_embedding, label)``

    Args:
        tokenizer (torch Tokenizer):
            the tokenizer of the source text
        vocabulary (torch Vocabulary):
            the vocubulary; it can be built of it can be
            given.

    """
    def __init__(self, tokenizer=None,
                 vocabulary=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer

        self.vocabulary = vocabulary
        self.MAX_LENGTH = 0
        self.is_fitted = False

    def fit_to_data(self, data):
        """Method to extract global data, like to length of
        the sentences to be able to pad.

        Args:
            data (iterable):
                the data in the format ``(label, text)``
        """

        counter = Counter()  # for the text
        for (label, text) in data:
            if isinstance(text, tuple) or isinstance(text, list):
                text = text[0]
            counter.update(self.tokenizer(text))
            self.MAX_LENGTH = max(self.MAX_LENGTH, len(self.tokenizer(text)))
        # self.vocabulary = Vocab(counter, min_freq=1)
        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        self.is_fitted = True
        self.save_pretrained(".")

    def __call__(self, datum: tuple) -> Tensor:
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
            _text = _text[0]
        processed_text = torch.tensor(text_pipeline(_text),
                                      dtype=torch.int64).to(DEVICE)
        # convert to tensors (padded)
        out = torch.cat([processed_text,
                   pad_item * torch.ones(self.MAX_LENGTH - processed_text.shape[0]
                                              ).to(DEVICE)])
        return out


class PreprocessTextLabel(AbstractPreprocessing):
    def __init__(self, tokenizer=None, **kwargs):
        pass

    def fit_to_data(self, dataset):
        pass

    def __call__(self, datum: Tensor) -> Tensor:
        label_pipeline = lambda x: torch.tensor(x, dtype=torch.long) - 1

        _label = datum
        try:
            label_pipeline(_label).to(DEVICE)
        except TypeError:
            if isinstance(_label, tuple) or isinstance(_label, list):
                _label = _label[0]
        out = label_pipeline(_label).to(DEVICE)

        return out


class PreprocessTextTranslation(AbstractPreprocessing):
    """Class to preprocess text dataloaders for translation
    tasks

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
        from gdeep.data import TextDatasetTranslation, PreprocessTextTranslation

        dl = TorchDataLoader(name="Multi30k", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDatasetTranslation(dl_tr.dataset,
            PreprocessTextTranslation(), None)

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
        self.MAX_LENGTH = 0
        self.is_fitted = False

    def fit_to_data(self, data):
        counter = Counter()  # for the text
        counter_target = Counter()  # for the text
        for text in data:
            counter.update(self.tokenizer(text[0]))
            counter_target.update(self.tokenizer_target(text[1]))
            self.MAX_LENGTH = max(self.MAX_LENGTH, len(self.tokenizer(text[0])))
            self.MAX_LENGTH = max(self.MAX_LENGTH, len(self.tokenizer_target(text[1])))
        # self.vocabulary = Vocab(counter, min_freq=1)
        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        if self.vocabulary_target is None:
            self.vocabulary_target = Vocab(counter_target)
        self.is_fitted = True
        self.save_pretrained(".")

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
                                      dtype=torch.int64).to(DEVICE)
        processed_text_target = torch.tensor(text_pipeline(datum[1]),
                                      dtype=torch.int64).to(DEVICE)
        # convert to tensors (padded)
        out = torch.cat([processed_text,
                         pad_item * torch.ones(self.MAX_LENGTH - processed_text.shape[0]
                                               ).to(DEVICE)])
        out_target = torch.cat([processed_text_target,
                         pad_item_target * torch.ones(self.MAX_LENGTH - processed_text_target.shape[0]
                                               ).to(DEVICE)])
        return out, out_target


class PreprocessTextQA(AbstractPreprocessing):
    """Class to preprocess text dataloaders for Q&A
    tasks

    Args:
        vocabulary (torch Vocab):
            the torch vocabulary
        tokenizer (torch Tokenizer):
            the tokenizer of the source text

    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import  TextDatasetQA, PreprocessTextQA, PreprocessTextQATarget

        dl = TorchDataLoader(name="SQuAD2", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TextDatasetQA(dl_tr_str.dataset,
                               PreprocessTextQA(),
                               PreprocessTextQATarget())

    """

    def __init__(self, vocabulary=None,
                 tokenizer=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.MAX_LENGTH = 0
        self.is_fitted = False

    def fit_to_data(self, data):

        counter = Counter()  # for the text
        for (context, question, answer, init_position) in data:
            if isinstance(context, tuple) or isinstance(context, list):
                context = context[0]
            counter.update(self.tokenizer(context))
            self.MAX_LENGTH = max(self.MAX_LENGTH, len(self.tokenizer(context)))
            if isinstance(question, tuple) or isinstance(question, list):
                question = question[0]
            counter.update(self.tokenizer(question))
            self.MAX_LENGTH = max(self.MAX_LENGTH, len(self.tokenizer(question)))
            #if isinstance(answer, tuple) or isinstance(answer, list):
            #    answer = answer[0]
            #    if isinstance(answer, tuple) or isinstance(answer, list):
            #        answer = answer[0]
            #counter.update(self.tokenizer(answer))
            #self.MAX_LENGTH_ANSWER = max(self.MAX_LENGTH_ANSWER, len(self.tokenizer(answer)))
        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        self.pad_item = self.vocabulary["."]
        self.is_fitted = False
        self.save_pretrained(".")

    def __call__(self, datum:tuple) -> Tuple[Tensor, Tensor]:
        if not self.is_fitted:
            self.load_pretrained(".")
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        processed_context = torch.tensor(text_pipeline(datum[0]),
                                      dtype=torch.int64).to(DEVICE)
        out_context = torch.cat([processed_context,
                         self.pad_item * torch.ones(self.MAX_LENGTH - processed_context.shape[0]
                                               ).to(DEVICE)])
        processed_question = torch.tensor(text_pipeline(datum[1]),
                                         dtype=torch.int64).to(DEVICE)

        out_question = torch.cat([processed_question,
                         self.pad_item * torch.ones(self.MAX_LENGTH - processed_question.shape[0]
                                               ).to(DEVICE)])

        return out_context, out_question


class PreprocessTextQATarget(AbstractPreprocessing):
    """Class to preprocess text dataloaders for Q&A
    tasks

        Args:
            tokenizer (torch Tokenizer):
                the tokenizer of the source text

    """

    def __init__(self, tokenizer=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        self.MAX_LENGTH = 0
        self.is_fitted = False

    def fit_to_data(self, data):
        pass

    def __call__(self, datum):

        pos_init_char = datum[3][0]
        pos_init = len(self.tokenizer(datum[0][:pos_init_char]))
        pos_end = pos_init + len(self.tokenizer(datum[2][0]))

        return torch.tensor(pos_init, dtype=torch.long), torch.tensor(pos_end, dtype=torch.long)


class PreprocessImageClassification(AbstractPreprocessing):
    """Class to preprocess image files for classification
      tasks

          Args:
              size (int or sequence):
                  Desired output size. If size is a sequence like (h, w),
                  output size will be matched to this. If size is an int,
                  smaller edge of the image will be matched to this number.
                  I.e, if height > width, then image will be rescaled to
                  ``(size * height / width, size)``.

      """
    def __init__(self, size: Union[int, tuple]) -> None:
        self.size = size

    def fit_to_data(self, dataset:Dataset) -> None:
        pass

    def __call__(self, datum: Tensor) -> Tensor:
        return ToTensor()(Resize(self.size)(datum))

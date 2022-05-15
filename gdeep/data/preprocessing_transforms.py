import warnings
from collections import Counter
from typing import Any, List, Optional, Tuple, TypeVar, Union

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchvision.transforms import Resize, ToTensor

from .abstract_preprocessing import AbstractPreprocessing
from .transforming_dataset import TransformingDataset

# type definition
Tensor = torch.Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


T = TypeVar('T')
class Normalisation(AbstractPreprocessing[Tuple[Tensor, T], Tuple[Tensor, T]]):
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

    def fit_to_dataset(self, dataset: Dataset[Tuple[Tensor, T]]) -> None:
        self.mean = _compute_mean_of_dataset(dataset)
        self.stddev = _compute_stddev_of_dataset(dataset, self.mean)
        self.is_fitted = True

    def __call__(self, item: Tuple[Tensor, T]) -> Tuple[Tensor, T]:
        if not self.is_fitted:
            raise RuntimeError("The normalisation is not fitted to any dataset."
                               " Please call fit_to_dataset() first.")
        if not torch.all(self.stddev > 0):
            warnings.warn("The standard deviation contains zeros! Adding 1e-7")
            self.stddev = self.stddev + 1e-7
        out = (item[0] - self.mean)/ self.stddev
        return (out, item[1])

def _compute_mean_of_dataset(dataset: Dataset[Tuple[Tensor, Any]]) -> Tensor:
    """Compute the mean of the whole dataset"""
    mean: Tensor = torch.zeros(dataset[0][0].shape, dtype=torch.float64, device=DEVICE)
    for idx in range(len(dataset)):  # type: ignore
        if idx == 0:
            mean += dataset[idx][0]
        else:
            mean = (mean * idx + dataset[idx][0]) / (idx + 1)
    return mean

def _compute_stddev_of_dataset(dataset: Dataset[Tuple[Tensor, Any]], mean: Tensor) -> Tensor:
    """Compute the stddev of the whole dataset"""
    mean_normalized_dataset = TransformingDataset(dataset, lambda x: (x - mean)**2)
    stddev: Tensor = _compute_mean_of_dataset(mean_normalized_dataset)
    return stddev.sqrt()


class PreprocessTextData(AbstractPreprocessing[str, Tensor]):
    """Preprocessing class. This class is useful to convert the data format
    ``(label, text)`` into the proper tensor format ``( word_embedding, label)``

    Args:
        tokenizer (torch Tokenizer):
            the tokenizer of the source text
        vocabulary (torch Vocabulary):
            the vocubulary; it can be built of it can be
            given.

    """
    tokenizer: Any
    vocabulary: Optional[Vocab]
    
    def __init__(self, 
                 tokenizer: Optional[Any] = None,
                 vocabulary: Optional[Vocab] = None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer

        self.vocabulary = vocabulary
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[str, Any]]) -> None:
        """Method to extract global data, like to length of
        the sentences to be able to pad.

        Args:
            data (iterable):
                the data in the format ``(label, text)``
        """

        counter: Counter[str] = Counter()  # for the text
        for _, text in dataset:
            if isinstance(text, tuple) or isinstance(text, list):
                text = text[0]
            counter.update(self.tokenizer(text))
            self.max_length = max(self.max_length, len(self.tokenizer(text)))
        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        self.is_fitted = True
        self.save_pretrained(".")

    def __call__(self, datum: str) -> Tensor:
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
                   pad_item * torch.ones(self.max_length - processed_text.shape[0]
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

    def __init__(self,
                 vocabulary: Optional[Vocab]=None,
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


class PreprocessTextQA(AbstractPreprocessing[Tuple[str, str, List[str], List[str]], Tuple[Tensor, Tensor]]):
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

    def __init__(self, 
                 vocabulary=None,
                 tokenizer=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[str, str, List[str], List[str]]]):

        counter = Counter()  # for the text
        for (context, question, answer, init_position) in dataset:
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

    def __call__(self, datum: Tuple[str, str, List[str], List[str]]) -> Tuple[Tensor, Tensor]:
        if not self.is_fitted:
            raise ValueError("You need to fit the preprocessing first"
                             " using the fit_to_dataset method")
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        processed_context = torch.tensor(text_pipeline(datum[0]),
                                      dtype=torch.int64).to(DEVICE)
        out_context = torch.cat([processed_context,
                         self.pad_item * torch.ones(self.max_length - processed_context.shape[0]
                                               )]).to(DEVICE)
        processed_question = torch.tensor(text_pipeline(datum[1]),
                                         dtype=torch.int64).to(DEVICE)

        out_question = torch.cat([processed_question,
                         self.pad_item * torch.ones(self.max_length - processed_question.shape[0]
                                               )]).to(DEVICE)

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


class PreprocessImageClassification(AbstractPreprocessing[Tensor, Any]):
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
    def __init__(self, size: Union[int, Tuple[int, ...]]) -> None:
        self.size = size

    def fit_to_dataset(self, dataset:Dataset[Tuple[Tensor, int]]) -> None:
        # This preprocessor does not need to be fit to the dataset
        pass

    def __call__(self, datum: Tensor) -> Tensor:
        return ToTensor()(Resize(self.size)(datum))  # type: ignore

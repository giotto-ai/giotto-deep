from collections import Counter, OrderedDict
from typing import Callable, Tuple, Union, Any, Optional, List, Dict
from functools import partial as Partial  # noqa

import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torchtext.vocab.vocab import Vocab

from gdeep.utility import DEVICE
from ..abstract_preprocessing import AbstractPreprocessing
from .._utils import MissingVocabularyError

# type definition
from gdeep.utility.custom_types import Tensor


class TokenizerTextClassification(
    AbstractPreprocessing[Tuple[Any, str], Tuple[Tensor, Tensor]]
):
    """Preprocessing class. This class is useful to convert the data format
    ``(label, text)`` into the proper tensor format ``( word_embedding, label)``.
    The labels should be integers; if they are string, they will be converted.

    Args:
        tokenizer :
            the tokenizer of the source text
        vocabulary :
            the vocubulary; it can be built or it can be
            given.

    """

    max_length: int
    is_fitted: bool
    vocabulary: Optional[Vocab]
    tokenizer: Optional[Partial]
    counter: Dict[str, int]
    counter_label: Dict[str, int]

    def __init__(
        self, tokenizer: Optional[Partial] = None, vocabulary: Optional[Vocab] = None
    ):
        if tokenizer is None:
            self.tokenizer = get_tokenizer("basic_english")
        else:
            self.tokenizer = tokenizer

        self.vocabulary = vocabulary
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[Any, str]]) -> None:
        """Method to extract global data, like to length of
        the sentences to be able to pad.

        Args:
            dataset :
                the data in the format ``(label, text)``

        """

        self.counter = Counter()  # for the text
        self.counter_label = Counter()  # for the text
        self.ordered_dict = OrderedDict()
        self.list_of_possible_labels: List[Any] = []
        for (label, text) in dataset:  # type: ignore
            # if isinstance(text, tuple) or isinstance(text, list):
            #    text = text[0]
            self.counter.update(self.tokenizer(text))  # type: ignore
            self.counter_label.update([label])
            self.max_length = max(self.max_length, len(self.tokenizer(text)))  # type: ignore
        # build the vocabulary
        if not self.vocabulary:
            self.ordered_dict = OrderedDict(
                sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
            )
            self.list_of_possible_labels = list(self.counter_label.keys())
            self.vocabulary = vocab(self.ordered_dict)
            unk_token = "<unk>"  # type: ignore
            if unk_token not in self.vocabulary:
                self.vocabulary.insert_token(unk_token, 0)
            self.vocabulary.set_default_index(self.vocabulary[unk_token])
        self.is_fitted = True
        # self.save_pretrained(".")

    def __call__(self, datum: Tuple[Any, str]) -> Tuple[Tensor, Tensor]:
        """This method is applied to each datum and
        transforms it following the rule below

        Args:
            datum (tuple):
                a single datum, being it a tuple
                with ``(label, text)``
        """
        # if not self.is_fitted:
        #    self.load_pretrained(".")
        if self.vocabulary:
            text_pipeline: Callable[[str], List[int]] = lambda x: [
                self.vocabulary[token] for token in self.tokenizer(x)  # type: ignore
            ]  # type: ignore
        else:
            raise MissingVocabularyError(
                "Please fit this preprocessor to initialise the vocabulary"
            )
        pad_item: int = 0

        _text = datum[1]
        # if isinstance(_text, tuple) or isinstance(_text, list):
        #    _text = _text[1]
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.long).to(DEVICE)
        # convert to tensors (padded)
        out_text = torch.cat(
            [
                processed_text,
                pad_item
                * torch.ones(self.max_length - processed_text.shape[0]).to(DEVICE),
            ]
        ).to(torch.long)
        # preprocess labels
        label_pipeline = lambda x: torch.tensor(x, dtype=torch.long)

        _label = datum[0]
        out_label = label_pipeline(self.list_of_possible_labels.index(_label)).to(
            DEVICE
        )
        return out_text, out_label


from collections import Counter, OrderedDict
from typing import Callable, Tuple, \
    Optional, List, Dict

import torch
from torchtext.data.utils import get_tokenizer    # type: ignore
from torchtext.vocab import vocab
from gdeep.utility import DEVICE

from ..abstract_preprocessing import AbstractPreprocessing
from .._utils import MissingVocabularyError

# type definition
Tensor = torch.Tensor



class TokenizerTranslation(AbstractPreprocessing[Tuple[str, str], 
                                                 Tuple[Tensor, Tensor]
                           ]):
    """Class to preprocess text dataloaders for translation
    tasks. The Dataset type is supposed to be ``(string, string)``.
    The padding item is supposed to be of index 0.

        Args:
            vocabulary :
                the vocubulary of the source text;
                it can be built automatically or it can be
                given.
            vocabulary_target :
                the vocubulary of the target text;
                it can be built automatically or it can be
                given.
            tokenizer:
                the tokenizer of the source text
            tokenizer_target:
                the tokenizer of the target text

    Examples::

        from gdeep.data import DatasetBuilder
        from gdeep.data import TransformingDataset
        from gdeep.data.preprocessors import TokenizerTranslation

        db = DatasetBuilder(name="Multi30k", convert_to_map_dataset=True)
        ds_tr, ds_val, _ = db.build()

        textds = TransformingDataset(ds_tr,
            TokenizerTranslation())

        """
    if_fitted: bool
    vocabulary: Optional[Dict[str, int]]
    vocabulary_target: Optional[Dict[str, int]]
    tokenizer: Optional[Callable[[str], List[str]]]
    tokenizer_target: Optional[Callable[[str], List[str]]]
    ordered_dict: Dict[str, int]
    ordered_dict_target: Dict[str, int]

    def __init__(self, vocabulary:Optional[Dict[str, int]]=None,
                 vocabulary_target:Optional[Dict[str, int]]=None,
                 tokenizer:Optional[Callable[[str],List[str]]]=None,
                 tokenizer_target:Optional[Callable[[str],List[str]]]=None):

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
        self.counter = Counter()
        self.counter_target = Counter()
        self.ordered_dict = OrderedDict()
        self.ordered_dict_target = OrderedDict()
        for text in data:
            self.counter.update(self.tokenizer(text[0]))
            self.counter_target.update(self.tokenizer_target(text[1]))
            self.max_length = max(self.max_length, len(self.tokenizer(text[0])))
            self.max_length = max(self.max_length, len(self.tokenizer_target(text[1])))
        # self.vocabulary = Vocab(counter, min_freq=1)
        if self.vocabulary is None:
            self.ordered_dict = OrderedDict(sorted(self.counter.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True))
            self.vocabulary = vocab(self.ordered_dict)
            unk_token = '<unk>'  # type: ignore
            if unk_token not in self.vocabulary: self.vocabulary.insert_token(unk_token, 0)
            self.vocabulary.set_default_index(self.vocabulary[unk_token])
        if self.vocabulary_target is None:
            self.ordered_dict_target = OrderedDict(sorted(self.counter_target.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True))
            self.vocabulary_target = vocab(self.ordered_dict_target)
            unk_token = '<unk>'  # type: ignore
            if unk_token not in self.vocabulary_target: self.vocabulary_target.insert_token(unk_token, 0)
            self.vocabulary_target.set_default_index(self.vocabulary_target[unk_token])
            #self.vocabulary_target = Vocab(self.counter_target)
        self.is_fitted = True
        #self.save_pretrained(".")

    def __call__(self, datum: Tuple[str, str]) -> Tuple[Tensor, Tensor]:
        """This method is applied to each item of
        the dataset and
        transforms it following the rule described
        in this method

        Args:
            datum (torch.tensor):
                a single datum
        """

        #if not self.is_fitted:
        #    self.load_pretrained(".")
        if self.vocabulary:
            text_pipeline: Callable[[str], List[int]] = lambda x: [self.vocabulary[token] for token in  # type: ignore
                                       self.tokenizer(x)]  # type: ignore
        else:
            raise MissingVocabularyError("Please fit this preprocessor to initialise the vocabulary")

        if self.vocabulary_target:
            text_pipeline_target: Callable[[str], List[int]] = lambda x: [self.vocabulary_target[token] for token in  # type: ignore
                                       self.tokenizer_target(x)]   # type: ignore
        else:
            raise MissingVocabularyError("Please fit this preprocessor to initialise the vocabulary")

        pad_item: int = 0
        pad_item_target: int = 0

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


import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from torchtext.data.utils import get_tokenizer
from torch.nn.functional import pad
from collections import Counter
from torchtext.vocab import Vocab
#from transformers.feature_extraction_sequence_utils import \
#     FeatureExtractionMixin
import warnings
import os
import json
import jsonpickle

Tensor = torch.Tensor

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class AbstractPreprocessing(ABC):
    """The abstract class to define the interface of preprocessing
    """
    @abstractmethod
    def transform(self, *args, **kwargs):
        """This method deals with datum-wise transformations. This
        method is called in the Datasets to transform the output
        of ``__getitem__``"""
        pass

    @abstractmethod
    def fit_to_data(self, *args, **kwargs):
        """This method deals with getting dataset-level information.
        """
        pass

    def save_pretrained(self, path):
        with open(os.path.join(path, self.__class__.__name__ + ".json"), "w") as outfile:
            whole_class = jsonpickle.encode(self)
            json.dump(whole_class, outfile)

    def load_pretrained(self, path):
        try:
            with open(os.path.join(path,self.__class__.__name__ + ".json"), "r") as infile:
                whole_class = json.load(infile)
                self = jsonpickle.decode(whole_class)
        except FileNotFoundError:
            warnings.warn("The transformation file does not exist; attempting to run"
                          " the transformation anyway...")

class Normalisation(AbstractPreprocessing):
    """This class runs the standard normalisation on all the dimensions of
    the input tensor. For example, in case of images where each item is of
    shape ``(BS, C, H, W)``, the average will and the standard deviations
    will be tensors of shape ``(C, H, W)``
    """
    is_fitted: bool
    mean: Tensor

    def __init__(self):
        self.is_fitted = False

    def fit_to_data(self, data: Tensor):
        self.mean = self._mean(data, 0, False)
        self.stddev = self._stddev(data, 0, False)
        self.is_fitted = True
        self.save_pretrained(".")

    def transform(self, batch: Tensor) -> Tensor:
        if not self.is_fitted:
            self.load_pretrained(".")
        if not all(self.stddev>0):
            warnings.warn("The standard deviation contains zeros! Adding 1e-7")
            self.stddev = self.stddev + 1e-7
        out = torch.stack([(batch[i] - self.mean)/(self.stddev) for i in range(batch.shape[0])])
        return out

    def _mean(self, data, dim, keep_dim):
        if isinstance(data, torch.utils.data.Dataset):
            data=next(iter(DataLoader(data, batch_size=len(data))))[0]
        return torch.mean(data.float(), dim, keep_dim)

    def _stddev(self, data, dim, keep_dim):
        if isinstance(data, torch.utils.data.Dataset):
            data=next(iter(DataLoader(data, batch_size=len(data))))[0]
        return torch.std(data.float(), dim, keep_dim)


class PreprocessingPipeline(AbstractPreprocessing):
    """class to compose preprocessing classes

    Args:
        list_of_cls (list):
            list of class instances
    """
    def __init__(self, list_of_preproc_and_datatypes):
        self.list_of_cls = list_of_preproc_and_datatypes
        
    def fit_to_data(self, data, **kwargs):
        for (preproc_cls, data_type_class) in self.list_of_cls:
            preproc_cls.fit_to_data(data, **kwargs)
            data = data_type_class(data, preproc_cls, **kwargs)

    def transform(self, batch):
        for (cls, dt) in self.list_of_cls:
            batch = cls.transform(batch)
        return batch

    def __len__(self) -> int:
        return len(self.list_of_cls)

    def __getitem__(self, index: int):
        return self.list_of_cls[index]

    def __iter__(self):
        return iter(self.list_of_cls)

    def __repr__(self) -> str:
        return f'Pipeline({self.list_of_cls})'

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
        kwargs (dict):
            keyword arguments for the ``Vocab``
    """
    def __init__(self, tokenizer=None,
                 vocabulary=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer

        self.vocabulary = vocabulary
        self.MAX_LENGTH = 0
        self.check_calibration = False

    def fit_to_data(self, data):
        """Method to extract global data, like to length of
        the sentences to be able to pad.

        Args:
            data (torch dataset):
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
        self.check_calibration = True
        self.save_pretrained(".")

    def transform(self, batch: torch.Tensor) -> list:
        """This method is applied to each batch and
        transforms it following the rule below

        Args:
            batch (torch.tensor):
                a minibatch
        """
        if not self.check_calibration:
            self.load_pretrained(".")
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        pad_item = self.vocabulary["."]

        _text = batch
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

    def transform(self, batch: torch.Tensor) -> torch.Tensor:
        label_pipeline = lambda x: torch.tensor(x, dtype=torch.long) - 1

        _label = batch
        try:
            label_pipeline(_label).to(DEVICE)
        except TypeError:
            if isinstance(_label, tuple) or isinstance(_label, list):
                _label = _label[0]
        out = label_pipeline(_label).to(DEVICE)

        return out


class PreprocessText:
    """Class to preprocess text dataloaders

    Args:
        dataloaders (string):
            train and test dataloaders. Labels are
            expected to be tensors
        tokenizer (torch Tokenizer):
            the tokenizer of the source text
        kwargs (dict):
            keyword arguments for the ``Vocab``
    """

    def __init__(self, dataloaders,
                 tokenizer=None, vocabulary=None,
                 **kwargs):
        self.dataloaders = (list(dataloaders[0]), list(dataloaders[1]))
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        counter = Counter()  # for the text
        for (label, text) in self.dataloaders[0]:
            if isinstance(text, tuple) or isinstance(text, list):
                text = text[0]
            counter.update(self.tokenizer(text))
        # self.vocabulary = Vocab(counter, min_freq=1)
        if vocabulary is None:
            self.vocabulary = Vocab(counter, **kwargs)
        else:
            self.vocabulary = vocabulary

    def _loop_over_dataloader(self, dl):
        """helper function for the creation of dataset"""

        label_pipeline = lambda x: int(x) - 1
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        label_list, text_list = [], []
        MAX_LENGTH = 0
        pad_item = self.vocabulary["."]

        for (_label, _text) in dl:
            try:
                label_list.append(label_pipeline(_label.to(DEVICE)))
            except TypeError:
                if isinstance(_label, tuple) or isinstance(_label, list):
                    _label = _label[0]
                label_list.append(label_pipeline(_label.to(DEVICE)))

            if isinstance(_text, tuple) or isinstance(_text, list):
                _text = _text[0]
            processed_text = torch.tensor(text_pipeline(_text),
                                          dtype=torch.int64).to(DEVICE)
            MAX_LENGTH = max(MAX_LENGTH, processed_text.shape[0])
            text_list.append(processed_text)
        # convert to tensors
        label_list = torch.tensor(label_list).to(DEVICE)
        text_list = torch.stack([torch.cat([item,
                                            pad_item *
                                            torch.ones(MAX_LENGTH -
                                                       item.shape[0]
                                                       ).to(DEVICE)])
                                 for item in text_list]).to(DEVICE)
        return text_list, label_list

    def _build_dataset(self):
        """private method to prepare the datasets"""

        text_list, label_list = self._loop_over_dataloader(self.dataloaders[0])
        self.training_data = TextDataset(text_list, label_list)
        text_list, label_list = self._loop_over_dataloader(self.dataloaders[1])
        self.test_data = TextDataset(text_list, label_list)

    def build_dataloaders(self, **kwargs) -> tuple:
        """This method return the dataloaders of the tokenised
        sentences, each converted to a list of integers via the
        vocabulary. Hence, data can thus be treated as point data.

        Args:
            kwargs (dict):
                keyword arguments to add to the DataLoaders
        Returns:
            (tuple):
                training_dataloader, test_dataloader
        """
        self._build_dataset()
        train_dataloader = DataLoader(self.training_data,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     **kwargs)
        return train_dataloader, test_dataloader

    def collate_fn(self, batch):
        label_list, text_list, offsets = [], [], [0]
        label_pipeline = lambda x: int(x) - 1
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text),
                                          dtype=torch.int64).to(DEVICE)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64).to(DEVICE)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(DEVICE)
        text_list = torch.cat(text_list).to(DEVICE)
        return text_list, label_list, offsets











class PreprocessTextTranslation(PreprocessText):
    """Class to preprocess text dataloaders for translation
    tasks

        Args:
            name (string):
                check the available datasets at
                https://pytorch.org/vision/stable/datasets.html
            n_pts (int):
                number of points in customly generated
                point clouds
            tokenizer (torch Tokenizer):
                the tokenizer of the source text
            tokenizer_lab (torch Tokenizer):
                the tokenizer of the target text
        """

    def __init__(self, dataloaders,
                 tokenizer=None,
                 tokenizer_lab=None):
        self.dataloaders = (list(dataloaders[0]), list(dataloaders[1]))
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        if tokenizer_lab is None:
            self.tokenizer_lab = get_tokenizer('basic_english')
        else:
            self.tokenizer_lab = tokenizer_lab
        counter = Counter()  # for the text
        counter_lab = Counter()  # for the labels
        for (label, text) in self.dataloaders[0]:
            if isinstance(text, tuple) or isinstance(text, list):
                text = text[0]
            counter.update(self.tokenizer(text))
            if isinstance(label, tuple) or isinstance(label, list):
                label = label[0]
            counter_lab.update(self.tokenizer_lab(label))
        # self.vocabulary = Vocab(counter, min_freq=1)
        self.vocabulary = Vocab(counter)
        self.vocabulary_lab = Vocab(counter_lab)

    def _loop_over_dataloader(self, dl):
        """helper function for the creation of dataset"""

        label_pipeline = lambda x: int(x) - 1
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]
        label_pipeline_str = lambda x: [self.vocabulary_lab[token] for token in
                                        self.tokenizer_lab(x)]

        label_list, text_list = [], []
        MAX_LENGTH = 0
        pad_item = self.vocabulary["."]
        pad_item_lab = self.vocabulary_lab["."]

        for (_label, _text) in dl:
            try:
                label_list.append(label_pipeline(_label))
            except TypeError:
                if isinstance(_label, tuple) or isinstance(_label, list):
                    _label = _label[0]
                processed_label = torch.tensor(label_pipeline_str(_label),
                                               dtype=torch.int64).to(DEVICE)
                MAX_LENGTH = max(MAX_LENGTH, processed_label.shape[0])
                label_list.append(processed_label)
            if isinstance(_text, tuple) or isinstance(_text, list):
                _text = _text[0]
            processed_text = torch.tensor(text_pipeline(_text),
                                          dtype=torch.int64).to(DEVICE)
            MAX_LENGTH = max(MAX_LENGTH, processed_text.shape[0])
            text_list.append(processed_text)
            # offset_list.append(processed_text.size(0))
        try:
            label_list = torch.tensor(label_list).to(DEVICE)
        except (TypeError, ValueError):
            label_list = torch.stack([torch.cat([item,
                                                 pad_item_lab *
                                                 torch.ones(MAX_LENGTH -
                                                            item.shape[0]
                                                            ).to(DEVICE)])
                                      for item in label_list]).to(DEVICE)
        text_list = torch.stack([torch.cat([item,
                                            pad_item *
                                            torch.ones(MAX_LENGTH -
                                                       item.shape[0]
                                                       ).to(DEVICE)])
                                 for item in text_list]).to(DEVICE)
        return text_list, label_list

    def build_dataloaders(self, **kwargs) -> tuple:
        """This method return the dataloaders of the tokenised
        sentences, each converted to a list of integers via the
        vocabulary. Hence, data can thus be treated as point data.

        Args:
            kwargs (dict):
                keyword arguments to add to the DataLoaders
        Returns:
            (tuple):
                training_dataloader, test_dataloader
        """
        text_list, label_list = self._loop_over_dataloader(self.dataloaders[0])
        self.training_data = TextDataset(torch.stack((text_list, label_list), dim=1), label_list)
        text_list, label_list = self._loop_over_dataloader(self.dataloaders[1])
        self.test_data = TextDataset(torch.stack((text_list, label_list), dim=1), label_list)

        train_dataloader = DataLoader(self.training_data,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     **kwargs)
        return train_dataloader, test_dataloader


class PreprocessTextQA(PreprocessText):
    """Class to preprocess text dataloaders for Q&A
    tasks

        Args:
            dataloaders (list):
                list of dataloaders, e.g. (train, test).
            tokenizer (torch Tokenizer):
                the tokenizer of the source text

    """

    def __init__(self, dataloaders,
                 tokenizer=None):
        self.dataloaders = (list(dataloaders[0]), list(dataloaders[1]))
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        counter = Counter()  # for the text
        for (context, question, answer, init_position) in self.dataloaders[0]:
            if isinstance(context, tuple) or isinstance(context, list):
                context = context[0]
            counter.update(self.tokenizer(context))
            if isinstance(question, tuple) or isinstance(question, list):
                question = question[0]
            counter.update(self.tokenizer(question))
            if isinstance(answer, tuple) or isinstance(answer, list):
                answer = answer[0]
                if isinstance(answer, tuple) or isinstance(answer, list):
                    answer = answer[0]
            counter.update(self.tokenizer(answer))
        # self.vocabulary = Vocab(counter, min_freq=1)
        self.vocabulary = Vocab(counter)

    def _add_to_list(self, _answer, text_pipeline, MAX_LENGTH, answer_list):
        """Adding an item to the list and making sure the item
        is in the right format"""
        if isinstance(_answer, tuple) or isinstance(_answer, list):
            _answer = _answer[0]
            if isinstance(_answer, tuple) or isinstance(_answer, list):
                _answer = _answer[0]
        processed_answer = torch.tensor(text_pipeline(_answer),
                                        dtype=torch.int64).to(DEVICE)
        MAX_LENGTH = max(MAX_LENGTH, processed_answer.shape[0])
        answer_list.append(processed_answer)
        return MAX_LENGTH

    def _convert_list_to_tensor(self, answer_list, MAX_LENGTH, pad_item):
        """convert to tensor a list by padding"""
        try:
            answer_tensor = torch.tensor(answer_list).to(DEVICE)
        except (TypeError, ValueError):
            answer_tensor = torch.stack([torch.cat([item,
                                                    pad_item *
                                                    torch.ones(MAX_LENGTH -
                                                               item.shape[0]
                                                               ).to(DEVICE)])
                                         for item in answer_list]).to(DEVICE)
        return answer_tensor

    def _convert_to_token_index(self, _pos, _context):
        """This private method converts the chartacter index to the token index"""
        if isinstance(_context, tuple) or isinstance(_context, list):
            _context = _context[0]

        if isinstance(_pos, tuple) or isinstance(_pos, list):
            _pos = _pos[0]
        return len(self.tokenizer(_context[:_pos]))

    def _loop_over_dataloader(self, dl):
        """helper function for the creation of dataset"""

        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]

        question_list, context_list, pos_init_list, pos_end_list = [], [], [], []
        max_length_0, max_length_1, max_length_2 = 0, 0, 1
        self.pad_item = self.vocabulary["."]
        for (_context, _question, _answer, _pos) in dl:
            max_length_0 = self._add_to_list(_context, text_pipeline, max_length_0, context_list)
            max_length_1 = self._add_to_list(_question, text_pipeline, max_length_1, question_list)
            pos_init_list.append(self._convert_to_token_index(_pos, _context))
            pos_end_list.append(self._convert_to_token_index(_pos[0] + len(_answer[0][0]), _context))

        context_list = self._convert_list_to_tensor(context_list, max_length_0, self.pad_item)
        question_list = self._convert_list_to_tensor(question_list, max_length_1, self.pad_item)
        pos_init_list = self._convert_list_to_tensor(pos_init_list, max_length_2, self.pad_item)
        pos_end_list = self._convert_list_to_tensor(pos_end_list, max_length_2, self.pad_item)

        return question_list, context_list, pos_init_list, pos_end_list

    def build_dataloaders(self, **kwargs) -> tuple:
        """This method return the dataloaders of the tokenised
        sentences, each converted to a list of integers via the
        vocabulary. Hence, data can thus be treated as point data.

        Args:
            kwargs (dict):
                keyword arguments to add to the DataLoaders
        Returns:
            (tuple):
                training_dataloader, test_dataloader
        """
        MAX_LENGTH = 0
        padding_fn = lambda x: pad(x, (0, MAX_LENGTH - x.shape[1]), 'constant', self.pad_item)
        question_list, context_list, pos_init_list, pos_end_list = self._loop_over_dataloader(self.dataloaders[0])
        MAX_LENGTH = max((context_list.shape[1], question_list.shape[1]))

        datum = torch.stack(list(map(padding_fn, (context_list, question_list))), dim=2)
        label = torch.stack((pos_init_list, pos_end_list), dim=1)
        self.training_data = TextDatasetQA(datum, label)
        question_list, context_list, pos_init_list, pos_end_list = self._loop_over_dataloader(self.dataloaders[1])
        MAX_LENGTH = max((context_list.shape[1], question_list.shape[1]))

        datum = torch.stack(list(map(padding_fn, (context_list, question_list))), dim=2)
        label = torch.stack((pos_init_list, pos_end_list), dim=1)
        # print(datum.shape)  # expected to be (n_samples, MAX_LENGHT, 3)
        self.test_data = TextDatasetQA(datum, label)

        train_dataloader = DataLoader(self.training_data,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     **kwargs)
        return train_dataloader, test_dataloader
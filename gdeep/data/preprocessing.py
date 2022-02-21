import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.functional import pad
from collections import Counter
from torchtext.vocab import Vocab

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class TextDataset(Dataset):
    """This class is the base class for the text-datasets

    Args:
        data (Tensor):
            tensor with first dimension
            the number of samples
        targets (list):
            list of labels
        transform (Callable):
            act on the single images
        target_transform (Callable):
            act on the single label
    """

    def __init__(self, data, targets,
                 transform=None,
                 target_transform=None,
                 pos_transform=None):
        self.targets = targets
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.pos_transform = pos_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        sample = [text.to(torch.long), label.to(torch.long)]
        return sample

class TextDatasetTranslation(TextDataset):
    """This class is the base class for the text-datasets

    Args:
        data (Tensor):
            tensor with first dimension
            the number of samples
        targets (list):
            list of labels
        transform (Callable):
            act on the single images
        target_transform (Callable):
            act on the single label
    """

    def __init__(self, *args, **kwargs):
        super(TextDatasetTranslation, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            text = self.transform(text[0]), self.transform(text[1])
        if self.target_transform:
            label = self.target_transform(label)
        sample = [(text[0].to(torch.long), text[1].to(torch.long)), label.to(torch.long)]
        return sample


class TextDatasetQA(TextDataset):
    """This class is the base class for the text-datasets

    Args:
        data (Tensor):
            tensor with first dimension
            the number of samples
        targets (list):
            list of labels
        transform (Callable):
            act on the context and question
        target_transform (Callable):
            act on the single label
    """

    def __init__(self, *args, **kwargs):
        super(TextDatasetQA, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        context, question = self.data[idx][:,0], self.data[idx][:,1]
        pos_init, pos_end = self.targets[idx][0], self.targets[idx][1]
        if self.transform:
            context = self.transform(context)
            question = self.transform(question)
        if self.pos_transform:
            pos_init = self.pos_transform(pos_init)
            pos_end = self.pos_transform(pos_end)
        #sample = [(context.to(torch.long), question.to(torch.long)), (pos, answer.to(torch.long))]
        sample = [torch.stack((context,question)).to(torch.long),
                  torch.stack((pos_init,pos_end)).to(torch.long)]
        return sample


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
                 tokenizer=None, **kwargs):
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
        self.vocabulary = Vocab(counter, **kwargs)


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
            pos_init_list.append(self._convert_to_token_index(_pos,_context))
            pos_end_list.append(self._convert_to_token_index(_pos[0]+len(_answer[0][0]), _context))

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
        padding_fn = lambda x: pad(x,(0,MAX_LENGTH-x.shape[1]),'constant',self.pad_item)
        question_list, context_list, pos_init_list, pos_end_list = self._loop_over_dataloader(self.dataloaders[0])
        MAX_LENGTH = max((context_list.shape[1], question_list.shape[1]))

        datum = torch.stack(list(map(padding_fn,(context_list, question_list))), dim=2)
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
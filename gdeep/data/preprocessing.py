import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
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
                 transform=None, target_transform=None):
        self.targets = targets
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

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


class PreprocessText:
    """Class to preprocess text dataloaders

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
        except TypeError:
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
                                      pin_memory=True,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     pin_memory=True,
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
    def __init__(self, *args, **kwargs):
        super(PreprocessTextTranslation, self).__init__(*args, **kwargs)

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
                                      pin_memory=True,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     pin_memory=True,
                                     **kwargs)
        return train_dataloader, test_dataloader

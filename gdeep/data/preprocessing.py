import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab


class TextDataset(Dataset):
    """This class is the base class for the tori-datasets

    Args:
        data (Tensor): tensor with first dimension
            the number of samples
        taregts (list): list of labels
        transform (Callable): act on the single images
        target_transform (Callable): act on the single label
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
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = [image.to(torch.long), label.to(torch.long)]
        return sample


class PreprocessText:
    """Class to preprocess text dataloaders

    Args:
        name (string): check the available datasets at
            https://pytorch.org/vision/stable/datasets.html
        n_pts (int): number of points in customly generated
            point clouds
    """
    def __init__(self, dataloaders, tokenizer=None):
        self.dataloaders = (list(dataloaders[0]), list(dataloaders[1]))
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        counter = Counter()
        for (label, text) in self.dataloaders[0]:
            if type(text) is tuple:
                text = text[0]
            counter.update(self.tokenizer(text))
        # self.vocabulary = Vocab(counter, min_freq=1)
        self.vocabulary = Vocab(counter)
        
    def build_new_dataloaders(self, **kwargs):
        """This method return the dataloaders of the tokenised
        sentences, each converted to a list of integers via the
        vocabulary. Hence, data can thus be treated as point data.

        Returns:
            (tuple): training_dataloader, test_dataloader
        """

        label_pipeline = lambda x: int(x) - 1
        text_pipeline = lambda x: [self.vocabulary[token] for token in
                                   self.tokenizer(x)]
        
        label_list, text_list = [], []
        MAX_LENGTH = 0
        pad_item = self.vocabulary["."]
        
        for (_label, _text) in self.dataloaders[0]:
            label_list.append(label_pipeline(_label))
            if type(_text) is tuple:
                _text = _text[0]
            processed_text = torch.tensor(text_pipeline(_text),
                                          dtype=torch.int64)
            MAX_LENGTH = max(MAX_LENGTH, processed_text.shape[0])
            text_list.append(processed_text)
            # offset_list.append(processed_text.size(0))

        label_list = torch.tensor(label_list)
        text_list = torch.stack([torch.cat([item,
                                            pad_item *
                                            torch.ones(MAX_LENGTH -
                                                       item.shape[0])])
                                 for item in text_list])

        # offset_list = torch.tensor(offset_list[:-1]).cumsum(dim=0)
        training_data = TextDataset(text_list, label_list)
        label_list, text_list = [], []

        for (_label, _text) in self.dataloaders[1]:
            if type(_text) is tuple:
                _text = _text[0]
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text),
                                          dtype=torch.int64)
            text_list.append(processed_text)
            # offset_list.append(processed_text.size(0))
        label_list = torch.tensor(label_list)
        text_list = torch.stack([torch.cat([item,
                                            pad_item *
                                            torch.ones(MAX_LENGTH -
                                            item.shape[0])])
                                            for item in text_list])

        test_data = TextDataset(text_list, label_list)
        train_dataloader = DataLoader(training_data,
                                      **kwargs)
        test_dataloader = DataLoader(test_data,
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
                                          dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, label_list, offsets


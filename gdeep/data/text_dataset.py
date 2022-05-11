import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

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
        transform (class instance):
            the instance of the class of preprocessing
        target_transform (Callable):
            the instance of the class of preprocessing
    """

    def __init__(self, dataset,
                 transform=None,  # Optional[...]
                 target_transform=None,
                 pos_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.pos_transform = pos_transform
        # initialise the preprocessing
        if self.transform:
            self.transform.fit_to_data(self.dataset)
        if self.target_transform:
            self.target_transform.fit_to_data(self.dataset)
        if self.pos_transform:
            self.pos_transform.fit_to_data(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.dataset[idx][1]
        label = self.dataset[idx][0]
        if self.transform:
            text = self.transform.transform(text)
            #if idx < 8: print(text)
        if self.target_transform:
            label = self.target_transform.transform(label)
        if isinstance(label, torch.Tensor):
            if text is None: 
                print(text, label, idx)
            sample = [text.to(torch.long), label.to(torch.long)]
        else:
            sample = [text.to(torch.long), torch.tensor(label, dtype=torch.long)]
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

    def __getitem__(self, idx):
        text0 = self.dataset[idx]
        if self.transform:
            text = self.transform.transform(text[0]), self.transform(text[1])
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

    def __getitem__(self, idx):
        context, question = self.data[idx][:,0], self.data[idx][:,1]
        pos_init, pos_end = self.targets[idx][0], self.targets[idx][1]
        if self.transform:
            context = self.transform.batch_transform(context)
            question = self.transform.batch_transform(question)
        if self.pos_transform:
            pos_init = self.pos_transform.batch_transform(pos_init)
            pos_end = self.pos_transform.batch_transform(pos_end)
        #sample = [(context.to(torch.long), question.to(torch.long)), (pos, answer.to(torch.long))]
        sample = [torch.stack((context,question)).to(torch.long),
                  torch.stack((pos_init,pos_end)).to(torch.long)]
        return sample


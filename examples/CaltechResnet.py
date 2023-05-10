import math
import torch

from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Lambda

class CaltechResnet(nn.Module):
    def __init__(self):
        super(CaltechResnet, self).__init__()
        self.flow = nn.Sequential(
            resnet50(weights=ResNet50_Weights.DEFAULT),
            nn.Linear(1000, 256)
        )

    def forward(self, x):
        return self.flow(x)
    
class GrayscaleToRgb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return torch.tensor([img[0].tolist()] * 3) if len(img) == 1 else img
    

CALTECH_IMG_TRANSFORM = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    GrayscaleToRgb() # Duplicate channel if image has only 1 (greyscale)
])

CALTECH_LABEL_TRANSFORM = Lambda(lambda y: torch.zeros(256, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

# Creates samplers for each ratio found in split_ratios from the given caltech dataset
def caltechSamplers(caltech_ds, split_ratios):
    ratio_sum = sum(split_ratios)
    if ratio_sum > 1 or ratio_sum < 0:
        return []
    if ratio_sum < 1:
        split_ratios.append(1. - ratio_sum)

    indices = caltechSplitIndices(caltech_ds, split_ratios)

    samplers = []
    for idx_list in indices:
        samplers.append(SubsetRandomSampler(idx_list))

    return samplers

def caltechSplitIndices(caltech_ds, split_ratios):
    
    size_per_cat = [0] * len(caltech_ds.categories)
    for cat in caltech_ds.y:
        size_per_cat[cat] += 1
    size_per_cat.pop() # Last category is junk

    indices = []
    for i in split_ratios:
        indices.append([])
    idx = 0
    #print("Starting split with ratios:")
    #for ratio in split_ratios:
    #    print(ratio)
    for cat in range(len(size_per_cat)):
        #print(f'{caltech_ds.categories[cat]}: {size_per_cat[cat]}')
        cat_item_nb = 0
        split = 0
        while split < len(split_ratios) - 1:
            nb_items = math.floor(size_per_cat[cat] * split_ratios[split])
            #print(f'\t{split}: {nb_items} elements')
            indices[split].extend(range(idx, idx + nb_items))
            cat_item_nb += nb_items
            idx += nb_items
            split += 1
        last_items = size_per_cat[cat] - cat_item_nb
        #print(f'\t{split}: {last_items} elements')
        indices[split].extend(range(idx, idx + last_items))
        idx += last_items

    return indices

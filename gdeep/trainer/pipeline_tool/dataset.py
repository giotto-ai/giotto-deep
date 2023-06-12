import torch

class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, size, input_shape, output_shape):
        self.size = size
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Génération de données aléatoires
        data = torch.randn(*self.input_shape)
        target = torch.randint(0, 2, self.output_shape, dtype=torch.long)
        return data, target
    


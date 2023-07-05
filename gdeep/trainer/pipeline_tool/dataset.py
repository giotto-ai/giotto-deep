import torch

class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, size, input_shape, output_shape):
        """Initialize a PipelineDataset object.
        
        :param size: Size of the dataset.
        :type size: int
        :param input_shape: Shape of the input data.
        :type input_shape: tuple
        :param output_shape: Shape of the output data.
        :type output_shape: tuple
        """
        self.size = size
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __len__(self):
        """Return the length of the dataset.
        
        :return: Length of the dataset.
        :rtype: int
        """
        return self.size

    def __getitem__(self, idx):
        """Get an item from the dataset at the given index.
        
        :param idx: Index of the item.
        :type idx: int
        :return: Tuple of input data and target data.
        :rtype: tuple
        """
        data = torch.randn(*self.input_shape)
        target = torch.randint(0, 2, self.output_shape, dtype=torch.long)
        return data, target


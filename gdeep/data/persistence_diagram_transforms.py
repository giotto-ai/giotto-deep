import torch
import torchvision.transforms as transforms

Module = torch.nn.Module

class NormalizePersistenceDiagram(Module):
    """
    Normalize persistence diagrams.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: The input tensor.
        """
        return (x - self.mean) / self.std
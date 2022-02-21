from ..persformer import Persformer

import torch

def test_persformer_output():
    """Test output size of the persformer.
    """
    batch_size = 10
    input_length = 15
    for dim_input, dim_output in [(1, 2), (2, 3), (4, 5)]:
        model = Persformer(dim_input=dim_input, dim_output=dim_output)
        x = torch.randn(batch_size, input_length, dim_input)
        assert model(x).shape[-1] == dim_output
    
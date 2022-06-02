import torch

Tensor = torch.Tensor


def accuracy(prediction: Tensor, y: Tensor) -> float:
    """This function computes the accuracy
    for the given prediction and expected
    output

    Args:
        prediction:
            the output of your model. If ``X`` is
            a tensor, then ``prediction = model(X)``
        y:
            the corresponding expected results

    Returns:
        float:
            the value of the accuracy
        """
    correct: float = 0.0
    try:
        correct += (prediction.argmax(1) == y).to(torch.float).sum().item()
    except RuntimeError:
        correct += (prediction.argmax(2) == y).to(torch.float).sum().item()

    return correct / y.shape[0] * 100

import torch

def causal_mask(size):
    """
    Define a casual mask for decoder.

    :param size: the sequence length.
    :return: a boolean mask.
    """
    # torch.triu() returns the upper triangular part of the matrix that represents the words that come after
    # The matrix consist of all 0 and only the upper triangular has 1 values. Therefore we capture the words that come after
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)

    # Transform matrix to have all 1 and only the upper triangular has 0 values
    return mask == 0
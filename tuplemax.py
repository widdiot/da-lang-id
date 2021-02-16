import torch


def tuplemax_loss(output, target):
    """
    pytorch implementation of triplet wise tuplemax loss
    https://arxiv.org/pdf/1811.12290.pdf
    output: (N,C)
    target: (N)
    """

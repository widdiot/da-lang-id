import torch


def tuplemax_loss(output, target):
    """
    pytorch implementation of triplet wise tuplemax loss
    https://arxiv.org/pdf/1811.12290.pdf
    output: (N,C)
    target: (N)
    """
    n = output.size()[0]
    c = output.size()[-1]
    assert c == 8, "tuplemax loss only implemented for (8) classes"
    mask = torch.ones_like(output).scatter_(1, target.unsqueeze(1), 0.)
    false_logits = output[mask.bool()].view(n, c - 1)
    true_logits = torch.gather(output, 1, index=target.unsqueeze(1))
    first = false_logits.repeat(1, 3)
    second = torch.cat((torch.roll(false_logits, 1, 1), torch.roll(false_logits, 2, 1), torch.roll(false_logits, 3, 1)), 1)
    zeroth = true_logits.repeat(1, 21)
    combinations = torch.cat((zeroth.unsqueeze(-1), first.unsqueeze(-1), second.unsqueeze(-1)), -1)
    l = torch.log(torch.exp(combinations[:, :, 0]) / (
            torch.exp(combinations[:, :, 0]) + torch.exp(combinations[:, :, 1]) + torch.exp(combinations[:, :, 2])))
    return l.mean(-1)


if __name__ == "__main__":
    output = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]])
    target = torch.tensor([0, 1])

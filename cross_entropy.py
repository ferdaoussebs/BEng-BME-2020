import torch
import torch.nn.functional as F
import torch.nn as nn


def cross_entropy(logits, target):
    prediction = nn.Softmax(dim=1)(logits)
    target = torch.squeeze(target).type(torch.long)
    loss = F.cross_entropy(prediction, target)

    return loss

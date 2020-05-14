import torch


def dice_loss(logits, target, smooth=1):
    prediction = torch.nn.Sigmoid()(logits)

    prediction = prediction.contiguous()
    target = target.contiguous()

    trocar = (target == 1).type(torch.int)
    background = (target == 0).type(torch.int)
    target = torch.cat((background, trocar), 1)

    numerator = prediction * target
    denominator = (prediction + target) + smooth
    loss = 1 - (2 * (numerator / denominator))

    return loss.mean()

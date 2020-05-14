import torch


def generalised_dice_loss(logits, target):
    prediction = torch.nn.Sigmoid()(logits)

    epsilon = 0.0001
    prediction = prediction.contiguous()
    target = target.contiguous()

    trocar = (target == 1).type(torch.int)
    background = (target == 0).type(torch.int)
    target = torch.cat((background, trocar), 1)

    weight = torch.sum(target, (0, 2, 3), keepdim=True)
    weight = torch.reciprocal(torch.pow(weight, 2) + epsilon)

    numerator = torch.sum((target * prediction) * weight)
    denominator = torch.sum(((target + prediction)) * weight)
    loss = 1 - (2 * (numerator / denominator))

    return loss.mean()

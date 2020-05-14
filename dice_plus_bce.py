from dice_loss import dice_loss
from cross_entropy import cross_entropy


def dice_plus_bce(logits, target):

    dice = dice_loss(logits, target)
    bce = cross_entropy(logits, target)

    return dice + bce

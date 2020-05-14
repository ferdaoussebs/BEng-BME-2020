from cross_entropy import cross_entropy
from generalised_dice import generalised_dice_loss


def generalised_plus_bce(logits, target):

    gene = generalised_dice_loss(logits, target)
    bce = cross_entropy(logits, target)

    return gene + bce

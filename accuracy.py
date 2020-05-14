def accuracy(logits, target):
    pred = logits[:, 1, ...]
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    return pred.eq(target).float().mean()

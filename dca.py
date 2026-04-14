import numpy as np

def net_benefit(y, prob, threshold):
    pred = prob >= threshold
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()

    n = len(y)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

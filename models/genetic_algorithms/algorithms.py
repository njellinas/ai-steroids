import numpy as np


def logistic(w, x):
    z = np.dot(w, x)
    p = 1.0 / (1.0 + np.exp(-z))  # sigmoid function (gives probability of going up)
    if p > 0.5:
        action = 1
    else:
        action = 0
    return action

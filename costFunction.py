import numpy as np
import pandas as pd
from sigmoid import sigmoid

def costFunction(theta, X_train, y_train):
    # compute X * theta
    h = sigmoid(X_train @ theta)

    # compute cost
    cost = np.sum(-1 *(y_train * np.log(h)) - (1 - y_train) * np.log(1 - h))/X_train.shape[0]

    return cost




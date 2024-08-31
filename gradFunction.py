import numpy as np
import pandas as pd
from sigmoid import sigmoid
from costFunction import costFunction

def gradFunction(theta, X_train, y_train):
    # compute X * theta
    h = sigmoid(X_train @ theta)

    # calculate error
    error = h - y_train

    # calculate gradient
    gradient = np.array((X_train.T @ error)/X_train.shape[0])

    # return gradient
    return gradient



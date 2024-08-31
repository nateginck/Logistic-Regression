import numpy as np
import pandas as pd

# computes closed-form solution to linear regression with X_train <- training data, Y_train <- response_data
def normalEqn(X_train, y_train):
    # use closed form equation for theta
    return np.linalg.pinv(X_train.transpose() @ X_train) @ X_train.transpose() @ y_train


if '__main__' == __name__:
    # enter toy dataset
    toy = np.array([[1, 1, 1, 8], [1, 2, 2, 6], [1, 3, 3, 4], [1, 4, 4, 2]])

    # print estimate for toy dataset
    theta = normalEqn(toy[:, 0:3], toy[:, 3])
    print(theta)
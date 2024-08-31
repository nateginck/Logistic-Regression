# Nathaniel Ginck: 4429600

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from sigmoid import sigmoid
from costFunction import costFunction
from gradFunction import gradFunction
from normalEqn import normalEqn

# load trading data
training_data_df = pd.read_csv('input/hw3_data1.txt', header=None)

# convert to np array
training_data = training_data_df.values
training_data = np.insert(training_data, 0, 1, axis=1)

# 1a: define X and y
X = training_data[:, :3]
y = training_data[:, 3]

# print size of X and y
print("Size of X: ", X.shape)
print("Size of y: ", y.shape)

# 1b: plot training data

# separate based on response
cat_0 = training_data_df[training_data_df.iloc[:, 2] == 0]
cat_1 = training_data_df[training_data_df.iloc[:, 2] == 1]

# plot 0s along x and y axis
plt.scatter(cat_0.iloc[:, 0], cat_0.iloc[:, 1], marker='o', c='yellow')
# plot 1s along x and y axis
plt.scatter(cat_1.iloc[:, 0], cat_1.iloc[:, 1], marker='+', c='black')
# add labels
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend(['Not Admitted', 'Admitted'], loc='upper right')
plt.savefig('output/ps3-1-b.png')
plt.close()

# 1c: randomly divide the data
np.random.shuffle(training_data)
split = int(training_data.shape[0] * 0.9)

# divide into train and test sets for X and y
X_train = training_data[:split, :-1]
X_test = training_data[split:, :-1]

y_train = training_data[:split, -1]
y_test = training_data[split:, -1]

# 1d: test sigmoid function

# create range of values from -15 to 15
z = np.arange(-15, 15.01, 0.01)

# call function
gz = sigmoid(z)

# plot gz vs z
plt.plot(z, gz)
plt.xlabel("z")
plt.ylabel("gz")
plt.savefig('output/ps3-1-d.png')
plt.close()

# 1e: consider toy function
X_toy = np.array([[1, 1, 0], [1, 1, 3], [1, 3, 1], [1, 3, 4]])
y_toy = np.array([[0], [1], [0], [1]])
theta_toy = np.array([[2], [0], [0]])

# calculate cost
cost = costFunction(theta_toy, X_toy, y_toy)
print("The cost J equals: ", cost)

dJ = gradFunction(theta_toy, X_toy, y_toy)
print("The gradient equals: ", dJ)

# 1f: optimize cost function

# define theta vector
theta = np.array([[0], [0], [0]])

# call fmin_bfg
estimates = scipy.optimize.fmin_bfgs(f=costFunction, fprime=gradFunction, x0=theta, args=(X_train, y_train))
parameters = np.array(estimates)

# calculate cost
cost_optimal = costFunction(estimates, X_train, y_train)

# print values
print("Optimal Parameters theta: ", estimates)
print("Optimal Cost: ", cost_optimal)

# 1g: graph decision boundary
xlin = np.linspace(30, 100, 1000)
# calculate line in terms of x2
x_values = (-estimates[0] - estimates[1] * xlin) / estimates[2]

# plot 0s along x and y axis
plt.scatter(cat_0.iloc[:, 0], cat_0.iloc[:, 1], marker='o', c='yellow')
# plot 1s along x and y axis
plt.scatter(cat_1.iloc[:, 0], cat_1.iloc[:, 1], marker='+', c='black')
# plot decision boundary
plt.plot(xlin, x_values)

# add labels
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend(['Not Admitted', 'Admitted'], loc='upper right')
plt.savefig('output/ps3-1-f.png')
plt.close()

# 1h: compute accuracy on test set
test_estimates = sigmoid(X_test @ parameters)

count = 0;
# count correct classifications
for i in range(test_estimates.shape[0]):
    if (test_estimates[i] > 0.5 and y_test[i] == 1) or (test_estimates[i] < 0.5 and y_test[i] == 0):
        count = count + 1

accuracy = count / test_estimates.shape[0]
print("Accuracy of testing model: ", accuracy)

# 1i: compute admission probability for student
X_student = np.array([1, 60, 65])
student_estimate = sigmoid(X_student @ parameters)

print(student_estimate)
print("The student should be admitted, because his probability of admission is: ", student_estimate, " which is > 0.5.")

# 2a: read in profit data
profit_data_df = pd.read_csv('input/hw3_data2.csv', header=None)

# convert to np array
profit_data = profit_data_df.values
profit_data = np.insert(profit_data, 0, 1, axis=1)

# split X and y
X_profit = profit_data[:, : -1]
y_profit = profit_data[:, 2]

# add a quadratic term to X_profit
X_profit_squared = np.power(X_profit[:, 1], 2)
X_profit = np.insert(X_profit, 2, X_profit_squared, axis=1)

# solve for normal equation
profit_eqn = normalEqn(X_profit, y_profit)
print(profit_eqn)

# 2b: calculate line
xlin_profit = np.linspace(500, 1000, num=1000)
y_hat = profit_eqn[0] + profit_eqn[1]*xlin_profit + profit_eqn[2]*(xlin_profit**2)

# plot line on scatter
plt.scatter(profit_data_df[0], profit_data_df[1], color='red', marker='o', label='fitted model')
plt.plot(xlin_profit, y_hat, label='training data')
plt.xlabel("population in thousands, n")
plt.ylabel("profit")
plt.legend(['fitted model', 'training data'], loc='upper left')
plt.savefig('output/ps3-2-b.png')
plt.close()

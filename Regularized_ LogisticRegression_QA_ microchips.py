import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

# load dataset
X_train, y_train = load_data("data/ex2data2.txt")

# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2')
# Set the x-axis label
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

'''Figure shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.'''
print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

'''While the feature mapping allows us to build a more expressive classifier, it is also more susceptible to overfitting.'''

# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, lambda_=1):
    m, n = X.shape
    Loss = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f = sigmoid(z)
        Loss = Loss - (y[i] * np.log(f) + (1 - y[i]) * np.log(1 - f))
    total_cost = Loss / m
    return total_cost


def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost_without_reg = compute_cost(X, y, w, b)
    reg_cost = 0.
    for j in range(n):
        reg_cost = reg_cost + (w[j] ** 2)

    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + (lambda_ / (2 * m)) * reg_cost
    return total_cost

#Graduient

def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = sigmoid(np.dot(X[i], w) + b) - y[i]
        dj_db = dj_db + sigmoid(np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + z_wb * X[i, j]

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (w[j] * lambda_ / m)

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing

# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ to 1 (you can try varying this)
lambda_ = 0.01;
# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                    compute_cost_reg, compute_gradient_reg,
                                    alpha, iterations, lambda_)

plot_decision_boundary(w, b, X_mapped, y_train)

#Prediction
def predict(X, w, b):
    # number of predicted samples
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        if sigmoid(np.dot(X[i], w) + b) >= 0.5:
            p[i] = 1
    return p

p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
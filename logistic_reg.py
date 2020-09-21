import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser(description='Process some features')
parser.add_argument('--save', action='store_true', dest='save',
                    help='whether to save figures or not')
args = parser.parse_args(sys.argv[1:])
b_save = args.save
print(b_save)

thresh = 1e-10
iters_bound = 1000000000
epochs_bound = 1000000

def log_likelihood (x, y, theta):
    m = y.shape[0]
    sigmoid = lambda x: (1/(1 + np.exp(-x)))
    h_theta = sigmoid(np.matmul(theta.T, x.T).T)
    return (np.sum(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta), axis=0))

def grad_ll (X, Y, theta):
    m = Y.shape[0]
    sigmoid = lambda x: (1/(1 + np.exp(-x)))
    h_theta = sigmoid(np.matmul(theta.T, X.T).T)
    diff = ((Y - h_theta).reshape((m, 1)) * (X))
    return 1/m*(np.sum(diff, axis=0)).reshape((theta.shape[0], 1))

def stop(av_error, last_error):
    # return (min(grad_error(X, Y, theta)) < thresh)
    return (abs(av_error - last_error) < thresh)

def stochastic_grad_descent (X, Y, theta0, batch_size=None, learn_rate=0.01):
    global iters_bound, theta_df
    m = Y.shape[0]
    if (batch_size is None):
        batch_size = m
    n_iters = 0
    n_epochs = 0
    last_ll = np.inf
    av_ll = 0
    n_batches = m/batch_size
    theta = theta0
    while ((not(stop(av_ll, last_ll))) and (n_epochs < epochs_bound)):
        last_ll = av_ll
        av_ll = 0
        for batch_num in range(0, m, batch_size):
            # if (stop(X, Y, theta, av_ll/(batch_num/batch_size))):
            #     return theta
            if (n_iters > iters_bound):
                return theta
            theta_df["theta0"].append(theta[0,0])
            theta_df["theta1"].append(theta[1,0])
            av_ll += np.asscalar(log_likelihood(X, Y, theta))
            theta += (learn_rate * grad_ll(X[batch_num:batch_num+batch_size, :], Y[batch_num:batch_num+batch_size, :], theta))
            n_iters += 1
        av_ll /= n_batches
        print(n_epochs, av_ll)
        n_epochs += 1
    return theta

def hessian(X, theta):
    m, n = X.shape
    fn = lambda x : (np.exp(-x)/((1 + np.exp(-x))**2))
    fi = fn(np.matmul(theta.T, X.T).T)
    hes_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hes_mat[i, j] = - np.sum(fi * X[:, i:i+1] * X[:, j:j+1], axis=0).item()
    return hes_mat

def newton_optimize(X, Y, theta0):
    n_iters = 0
    n_epochs = 0
    last_ll = np.inf
    av_ll = 0
    theta = theta0
    while ((not(stop(av_ll, last_ll))) and (n_epochs < epochs_bound)):
        last_ll = av_ll
        theta -= np.matmul(np.linalg.inv(hessian(X, theta)), grad_ll(X, Y, theta))
        av_ll = log_likelihood(X, Y, theta)
        # print(n_epochs, av_ll)
        n_epochs += 1
    return theta

X = np.array(pd.read_csv("data/q3/logisticX.csv", header=None))
Y = np.array(pd.read_csv("data/q3/logisticY.csv", header=None))
X = (X - X.mean(axis=0))/(np.std(X, axis=0))
X_ = np.insert(X, 0, 1, axis=1)
theta0 = np.zeros((X_.shape[1], 1))
theta_df = {"theta0":[], "theta1":[]}
# theta = stochastic_grad_descent(X_, Y, theta0)
theta = newton_optimize(X_, Y, theta0)
theta = theta.ravel()

print("Theta")
with np.printoptions(precision=3):
    print(theta)

y = Y.ravel()
X_0 = X[y == 0]
X_1 = X[y == 1]
plt.plot(X_0[:, 0], X_0[:, 1], 'bo', label=0)
plt.plot(X_1[:, 0], X_1[:, 1], 'r+', label=1)

x0s = np.linspace(np.min(X[:, 0]), np.max(X[:, 1]))
x1s = -1/(theta[2]) * (theta[0]  + theta[1] * x0s)
plt.plot(x0s, x1s, 'g-', label='decision boundary')
plt.xlabel('x0')
plt.ylabel('x1')
plt.legend(loc='best')
if (b_save):
    plt.savefig("Results/q3/classification.png")
else:
    plt.show()
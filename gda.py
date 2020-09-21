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

def estimate_mu0(X, Y, class0):
    return np.sum(np.where(Y == class0, X, 0), axis=0)/np.sum(np.where(Y == class0, 1, 0), axis=0).item()

def estimate_mu1(X, Y, class1):
    return np.sum(np.where(Y == class1, X, 0), axis=0)/np.sum(np.where(Y == class1, 1, 0), axis=0).item()

def estimate_var1(X, Y, class1):
    mu1 = estimate_mu1(X, Y, class1).reshape((X.shape[1], 1))
    var1 = np.zeros((2,2))
    n1 = 0
    for i in range(X.shape[0]):
        if (Y[i].item() == class1):
            var1 += np.matmul((X[i, :] - mu1), (X[i,:] - mu1).T)
            n1 += 1
    return var1/n1

def estimate_var0(X, Y, class0):
    mu0 = estimate_mu0(X, Y, class0).reshape((X.shape[1], 1))
    var0 = np.zeros((2,2))
    n0 = 0
    for i in range(X.shape[0]):
        if (Y[i].item() == class0):
            var0 += np.matmul((X[i, :] - mu0), (X[i,:] - mu0).T)
            n0 += 1
    return var0/n0

def linear_decision_boundary(X, Y, mu0, mu1, sigma):
    mu0 = mu0.reshape((mu0.shape[0], 1))
    mu1 = mu1.reshape((mu1.shape[0], 1))
    class0, class1 = np.unique(Y)
    m = Y.shape[0]
    phi = np.sum(np.where(Y == class0, 1, 0), axis=0).item()/m
    sigma_inv = np.linalg.inv(sigma)
    c = (np.log(phi/(1 - phi)) + 0.5*(np.matmul(mu1.T, np.matmul(sigma_inv, mu1)) - np.matmul(mu0.T, np.matmul(sigma_inv, mu0)))).item()
    m = np.matmul((mu0 - mu1).T, sigma_inv)
    return lambda x0: (- m[0, 0]/m[0, 1] * x0 - c/m[0, 1])

def solve_quad(a, b, c):
    # a x**2 + b x + c = 0
    return (-b + (b**2 - 4*a*c)**0.5)/(2*a)

def quadratic_decision_boundary(X, Y, mu0, mu1, sigma0, sigma1):
    mu0 = mu0.reshape((mu0.shape[0], 1))
    mu1 = mu1.reshape((mu1.shape[0], 1))
    class0, class1 = np.unique(Y)
    m = Y.shape[0]
    phi = np.sum(np.where(Y == class0, 1, 0), axis=0).item()/m
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    c0 = (2 * np.log(phi/(1 - phi)) + np.matmul(mu1.T, np.matmul(sigma1_inv, mu1)) - np.matmul(mu0.T, np.matmul(sigma0_inv, mu0))).item()
    A = sigma1_inv - sigma0_inv
    b0 = np.matmul(mu0.T, sigma0_inv) - np.matmul(mu1.T, sigma1_inv)
    b1 = np.matmul(sigma0_inv, mu0) - np.matmul(sigma1_inv, mu1)
    return lambda x0: solve_quad(A[1, 1], ((A[0, 1] + A[1, 0])*x0 + b0[0, 1] + b1[1, 0]), (c0 + A[0, 0]*(x0**2) + (b0[0, 0] + b1[0, 0])*x0))

X = np.array(pd.read_csv("data/q4/q4x.dat", sep=' ', header=None).dropna(axis=1))
Y = np.array(pd.read_csv("data/q4/q4y.dat", sep=' ', header=None))
classes = np.unique(Y)
mu0 = estimate_mu0(X, Y, classes[0])
mu1 = estimate_mu1(X, Y, classes[1])
var0 = estimate_var0(X, Y, classes[0])
var1 = estimate_var1(X, Y, classes[1])

with np.printoptions(precision=3):
    print("mu0: ")
    print(mu0)
    print("\n")
    print("mu1: ")
    print(mu1)
    print("\n")
    print("var0: ")
    print(var0)
    print("\n")
    print("var1: ")
    print(var1)
    print("\n")

y = Y.ravel()
X_0 = X[y == classes[0]]
X_1 = X[y == classes[1]]
plt.plot(X_0[:,0], X_0[:, 1], 'bo', label=classes[0])
plt.plot(X_1[:, 0], X_1[:, 1], 'r+', label=classes[1])
# Linear boundary
linear_db = linear_decision_boundary(X, Y, mu0, mu1, var0)
x0s = (X_0[:, 0], X_1[:, 0])
x0_db = np.linspace(np.min(np.concatenate(x0s)), np.max(np.concatenate(x0s)), 100)
x1_db = linear_db(x0_db)
plt.plot(x0_db, x1_db, 'g-', label='Linear boundary')
# Quadratic boundary
quadratic_db = quadratic_decision_boundary(X, Y, mu0, mu1, var0, var1)
x1_qdb = quadratic_db(x0_db)
plt.plot(x0_db, x1_qdb, 'y-', label='Quadratic boundary')
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Labelled Data")
plt.legend(loc='best')
if (b_save):
    plt.savefig("Results/q4/classification.png")
else:
    plt.show()
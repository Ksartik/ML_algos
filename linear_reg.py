import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import argparse 

parser = argparse.ArgumentParser(description='Process some features')
parser.add_argument('--save', action='store_true', dest='save',
                    help='whether to save figures or not')
args = parser.parse_args(sys.argv[1:])
b_save = args.save
print(b_save)

thresh = 1e-10
iters_bound = 100000
frame = 0

def error_theta (X, Y, theta0, theta1):
    m = Y.shape[0]
    return 1/(2*m)*(np.sum(np.square(Y - (theta0 + theta1*X)), axis=0).item())

def error (X, Y, theta):
    m = Y.shape[0]
    return 1/(2*m)*(np.sum(np.square(Y - np.matmul(theta.T, X.T).T), axis=0)).item()

def grad_error (X, Y, theta):
    m = Y.shape[0]
    diff = ((Y - np.matmul(theta.T, X.T).T) * (-X)).T
    return 1/m*(np.sum(diff, axis=1).reshape(theta.shape))

def stop(X, Y, theta, last_error):
    # return (min(grad_error(X, Y, theta)) < thresh)
    return (abs(error(X, Y, theta) - last_error) < thresh)

def grad_descent (X, Y, theta0, learn_rate=0.01):
    global iters_bound, theta_df
    theta = theta0
    n_iters = 0
    last_error = 0
    while ((not(stop(X, Y, theta, last_error))) and (n_iters < iters_bound)):
        theta_df["theta0"].append(theta[0,0])
        theta_df["theta1"].append(theta[1,0])
        last_error = error(X, Y, theta)
        theta -= learn_rate * grad_error(X, Y, theta)
        n_iters += 1
        # print(error(X, Y, theta))
    return theta

def update_lines(num, data, line) :
    global frame
    # print(data[:, frame])
    line.set_data(data[0:2, :frame])
    line.set_3d_properties(data[2, :frame])
    frame += 1

def animate_theta_error(thetas, fig=None, ax=None, title=None, save=False):
    global frame
    if ((fig is None) or (ax is None)):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
    if (title is None):
        title = '3D iteration theta'
    frame = 0
    data = []
    data.append(thetas[:, 0])
    data.append(thetas[:, 1])
    data.append(np.fromiter(map(lambda xi, yi: error_theta(X, Y, xi, yi), thetas[:, 0], thetas[:, 1]), dtype=np.float))
    data = np.array(data)
    print(data.shape)
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('theta0')
    ax.set_ylim3d([0.0, 0.1])
    ax.set_ylabel('theta1')
    ax.set_zlim3d([0.0, 0.5])
    ax.set_zlabel('Error')
    ax.set_title(title)
    line_ani = FuncAnimation(fig, update_lines, data.shape[1], fargs=(data, line), interval=200, blit=False)
    if (save):
        line_ani.save("Results/q1/" + title + ".gif", dpi=80, writer='imagemagick')
    else:
        plt.show()

def update_line2D(num, data, line) :
    global frame
    # print(data[:, frame])
    line.set_data(data[0:2, :frame])
    frame += 1

def animate_theta_error_2d(thetas, fig=None, ax=None, title="Theta iteration", save=False):
    global frame
    if ((fig is None) or (ax is None)):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
    frame = 0
    data = thetas.T
    line, = ax.plot(data[0, 0:1], data[1, 0:1])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('theta0')
    ax.set_ylim([0.0, 0.0025])
    ax.set_ylabel('theta1')
    ax.set_title(title)
    line_ani = FuncAnimation(fig, update_line2D, data.shape[1], fargs=(data, line), interval=200, blit=False)
    if (save):
        line_ani.save("Results/q1/" + title + ".gif", dpi=80, writer='imagemagick')
    else:
        plt.show()

def contour3d_animate (thetas, ngrid=100, save=False):
    x0 = np.linspace(np.min(thetas[:, 0]), np.max(thetas[:, 0]), num=ngrid)
    x1 = np.linspace(np.min(thetas[:, 1]), np.max(thetas[:, 1]), num=ngrid)
    x0, x1 = np.meshgrid(x0, x1)
    z = np.zeros(shape=(ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            z[i, j] = error_theta(X, Y, x0[i, j], x1[i, j])
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    contour = ax.plot_wireframe(x0, x1, z)
    animate_theta_error(thetas, fig=fig, ax=ax, save=save)

def contour_animate (thetas, ngrid=100, levels=30, title="Theta iteration", save=False):
    x0 = np.linspace(np.min(thetas[:, 0]), np.max(thetas[:, 0]), num=ngrid)
    x1 = np.linspace(np.min(thetas[:, 1]), np.max(thetas[:, 1]), num=ngrid)
    x0, x1 = np.meshgrid(x0, x1)
    z = np.zeros(shape=(ngrid, ngrid))
    for i in range(ngrid):
        for j in range(ngrid):
            z[i, j] = error_theta(X, Y, x0[i, j], x1[i, j])
    fig = plt.figure()
    ax = fig.gca()
    CS = ax.contour(x0, x1, z, levels)
    ax.axhline(0, color='black', alpha=.5, dashes=[2, 4],linewidth=1)
    ax.axvline(0, color='black', alpha=0.5, dashes=[2, 4],linewidth=1)
    plt.clabel(CS, inline=1, fontsize=8)
    animate_theta_error_2d (thetas, fig=fig, ax=ax, title=title, save=save)


np.set_printoptions(precision=3)

X = np.array(pd.read_csv("data/q1/linearX.csv", header=None))
Y = np.array(pd.read_csv("data/q1/linearY.csv", header=None))
X = (X - X.mean(axis=0))/(np.std(X, axis=0))
X_ = np.insert(X, 0, 1, axis=1)
theta_df = {"theta0": [], "theta1": []}
theta0 = np.zeros(shape=(X_.shape[1], 1))
theta = grad_descent(X_, Y, theta0)
print("Predicted theta:")
print(theta)

Xs, Ys = zip(*sorted(zip(X[:, 0], Y)))
Xs, Ys = np.array(Xs), np.array(Ys)
plt.plot(Xs, Ys, label='Actual')
plt.plot(Xs, theta[0] + theta[1] * Xs, label='Predicted')
plt.legend(loc = 'best')
plt.xlabel("Acidity")
plt.ylabel("Density")
plt.title("Comparison")
if (b_save):
    plt.savefig("Results/q1/comparison.png")
else:
    plt.show()

theta_df = pd.DataFrame(theta_df)
thetas = np.array(theta_df)

# 3D plot
contour3d_animate(thetas, save=b_save)

# 2D contour plots
contour_animate(thetas, title="Contour theta iteration (learn rate = 0.01)", save=b_save)

# 2D contour plots with different learning rate
theta_df = {"theta0": [], "theta1": []}
theta0 = np.zeros(shape=(X_.shape[1], 1))
theta = grad_descent(X_, Y, theta0, learn_rate=0.001)
thetas = np.array(pd.DataFrame(theta_df))
contour_animate(thetas, levels=30, title="Contour theta iteration (learn rate = 0.001)", save=b_save)

theta_df = {"theta0": [], "theta1": []}
theta0 = np.zeros(shape=(X_.shape[1], 1))
theta = grad_descent(X_, Y, theta0, learn_rate=0.025)
thetas = np.array(pd.DataFrame(theta_df))
contour_animate(thetas, levels=30, title="Contour theta iteration (learn rate = 0.025)", save=b_save)

theta_df = {"theta0": [], "theta1": []}
theta0 = np.zeros(shape=(X_.shape[1], 1))
theta = grad_descent(X_, Y, theta0, learn_rate=0.1)
thetas = np.array(pd.DataFrame(theta_df))
contour_animate(thetas, levels=30, title="Contour theta iteration (learn rate = 0.1)", save=b_save)

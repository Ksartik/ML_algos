import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3

thresh = 1e-3
thresh_strong = 1e-5
updates_bound = 1000000
epochs_bound = 10000
frame = 0
cycle_size = 1000

def error (X, Y, theta):
    m = Y.shape[0]
    return 1/(2*m)*(np.sum(np.square(Y - np.matmul(theta.T, X.T).T), axis=0)).item()

def grad_error (X, Y, theta):
    m = Y.shape[0]
    diff = ((Y - np.matmul(theta.T, X.T).T) * (-X)).T
    return 1/m*(np.sum(diff, axis=1).reshape(theta.shape))

def stop(av_error, last_error):
    # return (min(grad_error(X, Y, theta)) < thresh)
    return (abs(av_error - last_error) < thresh_strong)

def stop_cycle(av_error, last_error):
    # return (min(grad_error(X, Y, theta)) < thresh)
    return (abs(av_error - last_error) < thresh)

def stochastic_grad_descent (X, Y, theta0, batch_size=None, learn_rate=0.01):
    global updates_bound, theta_df
    m = Y.shape[0]
    if (batch_size is None):
        batch_size = m
    n_updates = 0
    n_epochs = 0
    last_error = np.inf
    av_error = 0
    n_batches = m/batch_size
    theta = theta0
    n_updates_cycle = 0
    av_cycle_error = 0
    last_cycle_error = 0
    while ((not(stop(av_error, last_error))) and (n_epochs < epochs_bound)):
        last_error = av_error
        av_error = 0
        for batch_num in range(0, m, batch_size):
            # if (stop(X, Y, theta, av_error/(batch_num/batch_size))):
            #     return theta
            if (n_updates_cycle > cycle_size):
                av_cycle_error /= cycle_size
                print(batch_num, av_cycle_error, last_cycle_error)
                if (stop_cycle (av_cycle_error, last_cycle_error)):
                    return theta
                last_cycle_error = av_cycle_error
                av_cycle_error = 0
                n_updates_cycle = 0
            if (n_updates > updates_bound):
                print(n_updates)
                return theta
            theta_df["theta0"].append(theta[0,0])
            theta_df["theta1"].append(theta[1,0])
            theta_df["theta2"].append(theta[2,0])
            av_error += error(X, Y, theta)
            av_cycle_error += error(X, Y, theta)
            theta -= (learn_rate * grad_error(X[batch_num:batch_num+batch_size, :], Y[batch_num:batch_num+batch_size], theta))
            # print(theta)
            n_updates += 1
            n_updates_cycle += 1
            # print(batch_num, av_error/((batch_num + 1)/batch_size))
        av_error /= n_batches
        n_epochs += 1
        if (n_epochs > epochs_bound):
            print(n_epochs)
        if (stop(av_error, last_error)):
            print(av_error, last_error)
    return theta

def update_lines(num, data, line) :
    global frame
    # print(data[:, frame])
    line.set_data(data[0:2, :frame])
    line.set_3d_properties(data[2, :frame])
    frame += 1

def plot_theta_movement(thetas, title="Iteration", xlimit=[0,1], ylimit=[0,1], zlimit=[0,1]):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    frame = 0
    data = thetas.T
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
    ax.set_xlim3d(xlimit)
    ax.set_xlabel('theta0')
    ax.set_ylim3d(ylimit)
    ax.set_ylabel('theta1')
    ax.set_zlim3d(zlimit)
    ax.set_zlabel('theta2')
    ax.set_title(title)
    line_ani = FuncAnimation(fig, update_lines, data.shape[1], fargs=(data, line), interval=200, blit=False)
    plt.show()  

N = 1000000
x1 = np.random.normal(3, 2, N)
x2 = np.random.normal(-1, 2, N)
epsilon = np.random.normal(0, 2**0.5, N).reshape((N, 1))
theta_orig = np.array([3, 1, 2]).reshape((3, 1))
x = np.insert(np.array([x1, x2]), 0, 1, axis=0).T
y = np.matmul(theta_orig.T, x.T).T + epsilon

np.set_printoptions(precision=3)

theta_df = {"theta0": [], "theta1": [], "theta2": []}
theta0 = np.zeros((3, 1), dtype=np.float)
theta_1 = stochastic_grad_descent(x, y, theta0, batch_size=1, learn_rate=0.0001)
thetas = np.array(pd.DataFrame(theta_df))
min_limits = np.min(thetas, axis=0)
max_limits = np.max(thetas, axis=0)
plot_theta_movement(thetas, title="Iteration (size = 1)", xlimit=[min_limits[0], max_limits[0]], ylimit=[min_limits[1], max_limits[1]], zlimit=[min_limits[2], max_limits[2]])
print("theta (batch_size = 1)")
print(theta_1)

theta_df = {"theta0": [], "theta1": [], "theta2": []}
theta0 = np.zeros((3, 1), dtype=np.float)
theta_2 = stochastic_grad_descent(x, y, theta0, batch_size=100, learn_rate=0.0001)
thetas = np.array(pd.DataFrame(theta_df))
min_limits = np.min(thetas, axis=0)
max_limits = np.max(thetas, axis=0)
plot_theta_movement(thetas, title="Iteration (size = 100)", xlimit=[min_limits[0], max_limits[0]], ylimit=[min_limits[1], max_limits[1]], zlimit=[min_limits[2], max_limits[2]])
print("theta (batch_size = 100)")
print(theta_2)

theta_df = {"theta0": [], "theta1": [], "theta2": []}
theta0 = np.zeros((3, 1), dtype=np.float)
theta_3 = stochastic_grad_descent(x, y, theta0, batch_size=10000, learn_rate=0.0001)
thetas = np.array(pd.DataFrame(theta_df))
min_limits = np.min(thetas, axis=0)
max_limits = np.max(thetas, axis=0)
plot_theta_movement(thetas, title="Iteration (size = 10000)", xlimit=[min_limits[0], max_limits[0]], ylimit=[min_limits[1], max_limits[1]], zlimit=[min_limits[2], max_limits[2]])
print("theta (batch_size = 10000)")
print(theta_3)

theta_df = {"theta0": [], "theta1": [], "theta2": []}
theta0 = np.zeros((3, 1), dtype=np.float)
theta_4 = stochastic_grad_descent(x, y, theta0, batch_size=1000000, learn_rate=0.0001)
thetas = np.array(pd.DataFrame(theta_df))
min_limits = np.min(thetas, axis=0)
max_limits = np.max(thetas, axis=0)
plot_theta_movement(thetas, title="Iteration (size = 1000000)", xlimit=[min_limits[0], max_limits[0]], ylimit=[min_limits[1], max_limits[1]], zlimit=[min_limits[2], max_limits[2]])
print("theta (batch_size = 1000000)")
print(theta_4)

# Test
test_df = pd.read_csv("data/q2/q2test.csv")
X_test = np.array(test_df[["X_1", "X_2"]])
Y_test = np.array(test_df[["Y"]])
X_test = np.insert(X_test, 0, 1, axis=1)
test_error1 = error(X_test, Y_test, theta_1)
print("test_error (batch size = 1): ", test_error1)
test_error2 = error(X_test, Y_test, theta_2)
print("test_error (batch size = 100): ", test_error2)
test_error3 = error(X_test, Y_test, theta_3)
print("test_error (batch size = 10000): ", test_error3)
test_error4 = error(X_test, Y_test, theta_4)
print("test_error (batch size = 1000000): ", test_error4)
test_error_orig = error(X_test, Y_test, theta_orig)
print("test error with original theta: ", test_error_orig)
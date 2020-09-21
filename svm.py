import numpy as np
from cvxopt import solvers, matrix
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
import pickle 

available_kernels = ["linear", "gaussian"]

def accuracy (true_labels, pred_labels):
    n = pred_labels.shape[0]
    nacc_labels = 0
    for i in range(n):
        if (true_labels[i] == pred_labels[i]):
            nacc_labels += 1
    return nacc_labels/n

def gaussian_kernel (X, gamma=0.05):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = scipy.exp(-pairwise_sq_dists * gamma)
    return K

def gaussian_kernel2 (X1, X2, gamma=0.05):
    pairwise_sq_dists = euclidean_distances(X1, X2)
    K = np.exp(-pairwise_sq_dists * gamma)
    return K

class SVM_bin_cvxopt:
    def __init__ (self, bin_labels, C=1.0, kernel="linear", gamma=0.05, histogram_split=True, epsilon=1e-10):
        self.sol = {}
        self.X = None
        self.y = None
        self.w = None
        self.b = None
        self.gamma = gamma
        if (kernel in available_kernels):
            self.kernel = kernel  
        else:
            raise Exception("kernel not found")  
        self.bin_labels = bin_labels
        self.C = C
    
    def train (self, train_data, train_labels, calc_wb=True):
        y_train = np.where(train_labels == self.bin_labels[0], 1, -1)
        self.X = train_data
        self.y = y_train
        m = y_train.shape[0]
        y_mat = np.diag(y_train)
        if (self.kernel == "linear"):
            yX = np.matmul(y_mat, train_data)
            P = matrix(np.matmul(yX, yX.T))
        elif (self.kernel == "gaussian"):
            P = matrix(np.matmul(y_mat, np.matmul(gaussian_kernel(train_data, self.gamma), y_mat.T)))
        q = matrix((-1) * np.ones(m))
        G = matrix(np.concatenate((np.diag(np.ones(shape=m)*(-1)), np.diag(np.ones(shape=m))), axis=0))
        h = np.zeros(shape=2*m)
        h[m:] = self.C
        h = matrix(h)
        A = matrix(np.array(y_train.T.reshape((1, m)), dtype=np.float))
        b = matrix(np.zeros(shape=1))
        self.sol = solvers.qp(P, q, G, h, A, b)
        if (calc_wb):
            w = self.get_primal_w()
            b = self.get_primal_b()
        
    def get_primal_w (self):
        if (self.w is None):
            alpha = self.sol['x']
            yX = np.matmul(np.diag(self.y), self.X)
            self.w = np.matmul(yX.T, alpha).ravel()
        return self.w

    def get_primal_b (self):
        if (self.b is None):
            if (self.w is None):
                w = self.get_primal_w()
            if (self.kernel == "gaussian"):
                alpha = np.array(self.sol['x']).ravel()
                alpha_y = np.multiply(self.y, alpha)
                norm_dist = np.matmul(alpha_y, gaussian_kernel(self.X, gamma=self.gamma))
            else:
                norm_dist = np.matmul(self.w.T, self.X.T).T.ravel()
            self.b = - ((np.max(norm_dist[self.y == -1]) + np.min(norm_dist[self.y == 1]))/2)
        return self.b

    def nSV (self):
        hist_w_neg = np.histogram(self.w[self.w < 0], bins=5)
        hist_w_pos = np.histogram(self.w[self.w > 0], bins=5)
        return ([np.sum(hist_w_pos[0][1:]), np.sum(hist_w_neg[0][:-1])])
        
    def sv (self):
        hist_w_neg = np.histogram(self.w[self.w < 0], bins=5)
        hist_w_pos = np.histogram(self.w[self.w > 0], bins=5)
        return ([self.w[self.w > hist_w_pos[1][1]], self.w[self.w < hist_w_neg[1][-2]]])

    def predict (self, x_test):
        if (self.kernel == "gaussian"):
            alpha = np.array(self.sol['x']).ravel()
            alpha_y = np.multiply(self.y, alpha)
            dec = np.matmul(alpha_y, gaussian_kernel2(self.X, x_test, gamma=self.gamma)) + self.b
        else:
            alpha = np.array(self.sol['x']).ravel()
            alpha_y = np.multiply(self.y, alpha)
            dec = np.matmul(alpha_y, np.matmul(self.X, x_test.T)) + self.b
            # dec = np.matmul(self.w, x_test.T) + self.b
        return (np.where(dec > 0, self.bin_labels[0], self.bin_labels[1]))

    def load_model (self, filename):
        self.w = np.load(filename + "_w.npy")
        self.b = np.load(filename + "_b.npy")

    def save_model (self, filename):
        np.save(filename + "_w.npy", self.w)
        np.save(filename + "_b.npy", self.b)

    def score (self, x_test):
        if (self.kernel == "gaussian"):
            alpha = np.array(self.sol['x']).ravel()
            alpha_y = np.multiply(self.y, alpha)
            dec = np.matmul(alpha_y, gaussian_kernel2(self.X, x_test, gamma=self.gamma)) + self.b
        else:
            dec = np.matmul(self.w, x_test.T) + self.b
        return np.where (dec > 0, (1/(1 + np.exp(-np.abs(dec)))), (1 - 1/(1 + np.exp(-np.abs(dec)))))

class SVM_multi_cvxopt:
    def __init__ (self, kernel="gaussian"):
        self.nclasses = 0
        self.labels = []
        self.classifiers = {}
        self.kernel = kernel

    def train (self, train_data, train_labels, C=1.0, gamma=0.05):
        self.labels = np.array(np.unique(train_labels), dtype=np.int64)
        self.nclasses = self.labels.size
        self.classifiers = {}
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):
                li = self.labels[i]
                lj = self.labels[j]
                bin_labels = [li, lj]
                print(bin_labels)
                labels = train_labels[np.isin(train_labels, bin_labels)]
                data = train_data[np.isin(train_labels, bin_labels), :]
                classifier_i_j = SVM_bin_cvxopt(bin_labels, C=C, kernel=self.kernel, gamma=gamma)
                classifier_i_j.train(data, labels)
                self.classifiers[(i, j)] = classifier_i_j
                print("Done: (", i, j, ")")

    def predict (self, data):
        nexamples = data.shape[0]
        votes = np.zeros(shape=(nexamples, self.nclasses))
        scores = np.zeros(shape=(nexamples, self.nclasses))
        pred_labels = np.zeros(shape=nexamples)
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):
                li, lj = self.labels[i], self.labels[j]
                pred_classes = self.classifiers[(i,j)].predict(data)
                pred_ind_classes = np.where(pred_classes == li, i, j)
                pred_scores = self.classifiers[(i,j)].score(data)
                for k in range(nexamples):
                    votes[k, pred_ind_classes[k]] += 1
                    scores[k, pred_ind_classes[k]] += pred_scores[k]
        for k in range(nexamples):
            for i in range(self.nclasses):
                scores[k, i] = votes[k, i] + scores[k, i]/votes[k, i] if votes[k,i] != 0 else 0
        return np.vectorize(lambda i: max(self.labels, key=lambda x: scores[i, x]))(range(scores.shape[0]))

    def load_model (self, filename):
        with open(filename + '.p', "wb") as f:
            pickle.load(self.classifiers, f)
        self.nclasses = len(self.classifiers.keys())

    def save_model (self, filename):
        with open(filename + '.p', "wb") as f:
            pickle.dump(self.classifiers, f)
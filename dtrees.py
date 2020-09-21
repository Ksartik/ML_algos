import numpy as np
from math import log2
from sklearn.ensemble import RandomForestClassifier

def accuracy(pred_y, actual_y):
    n = 0
    for py, ry in zip(pred_y, actual_y):
        if (py == ry):
            n += 1
    return float(n)/actual_y.shape[0]

def entropy (n_yis):
    n_y = np.sum(n_yis)
    ey = 0
    for n_yi in n_yis:
        p_yi = float(n_yi)/float(n_y)
        ey += -(p_yi * log2(p_yi))
    return ey

def mi_attribute (x, y):
    # MI
    yis, n_yis = np.unique(y, return_counts=True)
    ey = entropy(n_yis)
    xis, n_xis = np.unique(x, return_counts=True)
    n_x = np.sum(n_xis)
    ey_x = 0.0
    for xi, n_xi in zip(xis, n_xis):
        px = n_xi/n_x
        y_x = y[x == xi]
        y_xis, n_y_xis = np.unique(y_x, return_counts=True)
        ey_x += px * entropy(n_y_xis)
    return (ey - ey_x)

def nnodes(tree, max_depth):
    def count_nodes(tree, depth):
        if ((tree.leaf) or (depth == max_depth)):
            return 1
        else:
            count = 1
            for child in tree.children:
                count += count_nodes(child, depth+1)
            return count
    return count_nodes(tree.node, 0)

class DTreeNode:
    def __init__(self, x=None, y=None, val=None, sid=None):
        self.sid = sid
        self.x = x
        self.y = y
        self.children = []
        if (val is not None):
            # Leaf
            self.leaf = True
            self.val = val
        else:
            # Non-leaf
            self.leaf = False
            self.val = None
        self.chosen_j = -1
        self.mid = None

    def set_max_val (self):
        uniqs, counts = np.unique(self.y, return_counts=True)
        self.val = uniqs[np.argmax(counts)]

    def best_attribute (self):
        if (not (self.leaf)):
            max_mi, max_j = (-np.inf), -1
            for j in range(self.x.shape[1]):
                # over all attributes
                mi_j = mi_attribute(self.x[:, j], self.y)
                if (max_mi < mi_j):
                    max_mi = mi_j
                    max_j = j
            return max_j
        else:
            return -1

    def grow (self):
        if (not(self.leaf)):
            unique_vals = np.unique(self.y)
            if (unique_vals.size == 1):
                self.val = unique_vals.item()
                self.leaf = True
                self.children = []
            else:
                best_x_j = self.best_attribute()
                self.chosen_j = best_x_j
                # Only here assumed that it is a median split
                med_j = np.median(self.x[:, best_x_j])
                self.mid = med_j
                left_indices = self.x[:, best_x_j] <= med_j
                right_indices = self.x[:, best_x_j] > med_j
                if (np.sum(left_indices) != 0):
                    lchild = DTreeNode(self.x[left_indices, :], self.y[left_indices])
                else:
                    lchild = DTreeNode(val=np.random.randint(2))
                if (np.sum(right_indices) != 0):
                    rchild = DTreeNode(self.x[right_indices, :], self.y[right_indices])
                else:
                    rchild = DTreeNode(val=np.random.randint(2))
                self.children = [lchild, rchild]
                # For prediction
                self.set_max_val()
            return self.children
        else:
            return []


class DTree :
    """
    A boolean function is to be represented here. 
    """
    def __init__(self, data, labels):
        self.node = DTreeNode(data, labels, sid=0)
        self.nnodes = 1

    def name_nodes(self):
        nodes = [self.node]
        curr_sid = 0
        while (len(nodes) > 0):
            new_nodes = []
            for node in nodes:
                node.sid = curr_sid
                curr_sid += 1
                new_nodes += node.children
            nodes = new_nodes
        self.nnodes = curr_sid

    def grow_tree(self, max_depth):
        depth = 1
        nodes = [self.node]
        while ((depth < max_depth) and (len(nodes) > 0)):
            new_nodes = []
            for node in nodes:
                new_nodes += node.grow()
            nodes = new_nodes
            depth += 1
        for node in nodes:
            # here also assumed that it is a boolean
            node.grow()
            if (not(node.leaf)):
                new_children = []
                for child in node.children:
                    if (child.leaf):
                        new_children.append(child)
                    else:
                        child.set_max_val()
                        new_children.append(DTreeNode(val=child.val, sid=child.sid))
                node.children = new_children
        self.name_nodes()

    def predict (self, xdata):
        def pred_rec_node (node, x):
            if (node.leaf):
                return node.val
            else:
                if (x[node.chosen_j] > node.mid):
                    return pred_rec_node(node.children[1], x)
                else:
                    return pred_rec_node(node.children[0], x)
        
        pred_y = np.zeros(shape=xdata.shape[0])
        for i in range(xdata.shape[0]):
            pred_y[i] = pred_rec_node(self.node, xdata[i, :])
        return pred_y

    def predict_depth (self, xdata, max_depth):
        def pred_rec_node (node, x, depth):
            if (node.leaf):
                return node.val
            elif (depth == max_depth):
                if (x[node.chosen_j] > node.mid):
                    child = node.children[1]
                else:
                    child = node.children[0]
                return child.val
            else:
                if (x[node.chosen_j] > node.mid):
                    return pred_rec_node(node.children[1], x, depth+1)
                else:
                    return pred_rec_node(node.children[0], x, depth+1)
        
        pred_y = np.zeros(shape=xdata.shape[0])
        for i in range(xdata.shape[0]):
            pred_y[i] = pred_rec_node(self.node, xdata[i, :], 1)
        return pred_y

    def remove_node(self, node_sid):
        def node_rm (node):
            if (node.sid == node_sid):
                node.children = [DTreeNode(val=node.children[0].val), DTreeNode(val=node.children[1].val)]
            else:
                for child in node.children:
                    node_rm(child)
        return node_rm(self.node)

    def prune(self, val_x, val_y):
        def pred_rec_node (node, x, rm_sid):
            if (node.leaf):
                return node.val
            elif (node.sid == rm_sid):
                if (x[node.chosen_j] > node.mid):
                    child = node.children[1]
                else:
                    child = node.children[0]
                return child.val
            else:
                if (x[node.chosen_j] > node.mid):
                    return pred_rec_node(node.children[1], x, rm_sid)
                else:
                    return pred_rec_node(node.children[0], x, rm_sid)
        
        nodes = [self.node]
        max_acc = accuracy(self.predict(val_x), val_y)
        max_sid = 0
        neg = 0
        while ((len(nodes) > 0) & (neg <= 5)):
            new_nodes = []
            max_sid = -1
            for node in nodes:
                pred_y = np.zeros(shape=val_x.shape[0])
                for j in range(pred_y.shape[0]):
                    pred_y[j] = pred_rec_node(self.node, val_x[j, :], node.sid)
                acc = accuracy(pred_y, val_y)
                # print(node.sid, acc)
                if (acc > max_acc):
                    max_acc = acc
                    max_sid = node.sid
            print(max_sid, neg)
            if (max_sid == -1):
                neg += 1
            else:
                self.remove_node(max_sid)
            for node in nodes:
                new_nodes += node.children
            nodes = new_nodes

    def print_tree (self):
        def rec_node (node, level=0):
            if (node.leaf):
                ret = "\t"*level + "y = " + str(node.val) + "\n"
            else:
                ret = "\t"*level + "x_" + str(node.chosen_j) + " <= " + str(int(node.mid)) + "\n"
            for child in node.children:
                ret += rec_node(child, level+1)
            return ret
        print (rec_node(self.node))

class RandomForestClf(RandomForestClassifier):
    def __init__(self, n_jobs=None, oob_score=False, n_estimators=100, min_samples_split=2, max_features="auto"):
        super().__init__(n_jobs=n_jobs, oob_score=oob_score, n_estimators=n_estimators,
                        min_samples_split=min_samples_split, max_features=max_features)
    
    def fit (self, X, y):
        super().fit(X, y)

    def score (self, X, y):
        return self.oob_score_
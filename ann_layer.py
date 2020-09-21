import numpy as np

def get_activation_fn(fn_name):
    if (fn_name == "sigmoid"):
        return lambda x: (1/(1 + np.exp(-x)))
    elif (fn_name == "relu"):
        return lambda x: np.where(x < 0, 0, x)
    elif (fn_name == "leaky-relu"):
        return lambda x: np.where(x > 0, x, 0.01*x)
    else:
        return lambda x: x

def get_this_initializer(init, initializer, fin, fout):
    if (init == "he-uniform"):
        return lambda: initializer(fin)
    elif (init == "xavier-uniform" or init == "xavier-normal"):
        return lambda: initializer(fin, fout)
    else:
        return initializer

def get_general_initializer(init, meta_params):
    if (init is None):
        return lambda: 0
    elif (init == "rand-uniform"):
        return lambda : np.random.uniform(meta_params['low'] if 'low' in meta_params else -1, meta_params['high'] if 'high' in meta_params else 1)
    elif (init == "rand-normal"):
        return lambda : np.random.normal(meta_params['mu'] if 'mu' in meta_params else 0, meta_params['sigma'] if 'sigma' in meta_params else 1)
    elif (init == "xavier-uniform"):
        return lambda fin, fout: np.random.uniform(-(6/(fin + fout))**0.5, (6/(fin + fout))**0.5)
    elif (init == "xavier-normal"):
        return lambda fin, fout: np.random.normal(0, (2/(fin + fout))**0.5)
    elif (init == "he-uniform"):
        return lambda fin: np.random.uniform(-(6/fin)**0.5, (6/fin)**0.5)
    elif (init == "he-normal"):
        return lambda fin: np.random.normal(0, (2/fin)**0.5)
    elif (init == "lecun-normal"):
        return lambda fin: np.random.normal(0, (1/fin)**0.5)


class Layer:
    def __init__ (self, this_nnodes, prev_nnodes, prev_layer, mini_batch_size, initializer=lambda:0, activation_fn_name="sigmoid"):
        self.inputs = np.zeros(shape=(mini_batch_size, prev_nnodes))
        self.z = np.zeros(shape=(mini_batch_size, this_nnodes))
        self.outputs = np.zeros(shape=(mini_batch_size, this_nnodes))
        self.next = None
        self.delta = np.zeros(shape=(mini_batch_size, this_nnodes))
        self.delo_delnetj = np.zeros(shape=(mini_batch_size, this_nnodes))
        if (prev_layer is not None):
            self.prev = prev_layer
            self.theta = np.zeros(shape=(prev_nnodes, this_nnodes))
            for i in range(prev_nnodes):
                for j in range(this_nnodes):
                    self.theta[i,j] = initializer()
            self.grad = np.zeros(shape=(prev_nnodes, this_nnodes))
        else:
            self.prev = None
        self.activation_fn_name = activation_fn_name
        self.activation = get_activation_fn(activation_fn_name)

    def __set_delo_delnetj(self):
        if (self.activation_fn_name == "sigmoid"):
            self.delo_delnetj = self.outputs * (1 - self.outputs)
        elif (self.activation_fn_name == "relu"):
            self.delo_delnetj = np.where(self.z < 0, 0, 1)
        elif (self.activation_fn_name == "leaky-relu"):
            self.delo_delnetj = np.where(self.z < 0, 0.01, 1)
        else:
            self.delo_delnetj = self.outputs

    def set_output (self, inputs):
        self.inputs = inputs
        if (self.prev is None):
            self.z = self.inputs
            self.outputs = self.inputs
        else:
            self.z = np.matmul(self.inputs, self.theta)
            self.outputs = self.activation(self.z)
            self.__set_delo_delnetj()
        return self.outputs
    
    def set_grad (self, y):
        if (self.prev is not None):
            if (self.next is None):
                # Output layer
                self.delta = (y - self.outputs) * self.delo_delnetj
            else:
                # removing the bias term from the next layer (1:, :)
                self.delta = np.matmul(self.next.delta, self.next.theta[1:, :].T) * self.delo_delnetj
            self.grad = - np.matmul(self.inputs.T, self.delta) / (y.shape[0])

    
class ANN:
    def __init__(self, n_features, n_classes, hidden_layers=[100], hidden_activation_fn='sigmoid', output_activation_fn='sigmoid', init=None, mini_batch_size=100, adaptive=False, learn_rate=0.01, meta_params={}):
        initializer = get_general_initializer(init, meta_params)
        n_layers = len(hidden_layers) + 2
        n_nodes = [n_features, n_features] + hidden_layers + [n_classes]
        # +1 for bias term
        layers = [Layer(n_nodes[1], n_nodes[0]+1, None, mini_batch_size)]
        for i in range(2, n_layers):
            layers.append(Layer(n_nodes[i], n_nodes[i-1]+1, layers[i-2], mini_batch_size, 
                            activation_fn_name=hidden_activation_fn, initializer=get_this_initializer(init, initializer, n_nodes[i-1], n_nodes[i+1])))
        layers.append(Layer(n_nodes[i+1], n_nodes[i]+1, layers[i-1], mini_batch_size, activation_fn_name=output_activation_fn, 
                            initializer=get_this_initializer(init, initializer, n_nodes[i], 1)))
        layers[i].next = None
        i -= 1
        while (i >= 0):
            layers[i].next = layers[i+1]
            i -= 1
        self.n_classes = n_classes
        self.n_features = n_features
        self.layers = np.array(layers)
        self.mini_batch_size = mini_batch_size
        self.adaptive = adaptive
        self.learn_rate = learn_rate

    def forward_propagate (self, inputs):
        # first column is for the bias
        for layer in self.layers:
            inputs = np.insert(layer.set_output(inputs), 0, 1, axis=1)
        return inputs[:, 1:]

    def backward_propagate (self, y):
        for layer in self.layers[:0:-1]:
            layer.set_grad(y)

    def grad_error (self, xb, yb):
        final_outputs = self.forward_propagate(xb)
        self.backward_propagate(yb)
        return np.sum(np.square((yb - final_outputs)))

    def update_weights (self, n_epochs, n_batches):
        eta = self.learn_rate(n_epochs, n_batches) if (self.adaptive) else self.learn_rate
        # except the first layer
        for layer in self.layers[1:]:
            layer.theta -= eta * layer.grad

    def train_mini_batch (self, x, y, epochs_bound=1000, thresh=None):
        m = x.shape[0]
        n_epochs = 0
        n_batches = m/self.mini_batch_size
        examples = np.arange(m)
        last_error = 0
        while (True):
            np.random.shuffle(examples)
            av_error = 0
            n_batches = 0
            for batch_off in range(0, m, self.mini_batch_size):
                b_examples = examples[batch_off:batch_off+self.mini_batch_size]
                xb = x[b_examples, :]
                yb = y[b_examples, :]
                b_error = self.grad_error(xb, yb)
                av_error += b_error
                self.update_weights(n_epochs+1, n_batches+1)
                n_batches += 1
            av_error /= 2*m
            n_epochs += 1
            print(n_epochs, av_error)
            if (n_epochs >= epochs_bound):
                self.n_epochs = n_epochs
                self.av_error = av_error
                return
            if (thresh is not None):
                if (abs(av_error - last_error) < thresh):
                    self.av_error = av_error
                    self.n_epochs = n_epochs
                    return
                else:
                    last_error = av_error
    
    def train_predict_mini_batch(self, x, y, thresh = 1e-3, epochs_bound = 10000):
        self.train_mini_batch(x, y, thresh=thresh, epochs_bound=epochs_bound)
        return (self.predict(x))

    def predict_probs(self, x):
        return self.forward_propagate(x)

    def predict(self, x):
        pred = np.zeros(shape=(x.shape[0], self.n_classes))
        outputs = self.forward_propagate(x)
        for i in range(pred.shape[0]):
            pred[i, np.argmax(outputs[i, :])] = 1
        return pred

    def score (self, x, y):
        pred = self.predict(x)
        acc_preds = 0
        for i in range(y.shape[0]):
            # Assuming 1-hot
            if (np.all(pred[i, :] == y[i, :])):
                acc_preds += 1
        return acc_preds/y.shape[0]
import numpy as np
from scipy.special import expit
import sys

class MNIST_NeuralNetwork:
    def __init__(self, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0,decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = 10
        self.n_input = 784
        self.n_hidden = 50
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.minibatches = minibatches

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size = self.n_hidden * (self.n_input + 1))
        w1 = w1.reshape(self.n_hidden, self.n_input + 1)

        w2 = np.random.uniform(-1.0, 1.0, size = self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)

        return w1, w2

    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _sigmoid(self, z):
        return expit(z)

    def _sigmoid_gradient(self, z):
        return self._sigmoid(z) * (1.0 - self._sigmoid(z))

    def _add_bias_unit(self, x, how='column'):
        if how == 'column':
            x_with_bias = np.ones((x.shape[0], x.shape[1] + 1))
            x_with_bias[:, 1:] = x
        elif how == 'row':
            x_with_bias = np.ones((x.shape[0] + 1, x.shape[1]))
            x_with_bias[1:, :] = x
        else:
            raise AttributeError('引数howは「column」または「row」を指定してください')

        return x_with_bias

    def _feedforward(self, x, w1, w2):
        a0 = self._add_bias_unit(x, how='column')
        z1 = w1.dot(a0.T)
        a1 = self._sigmoid(z1) 
        a1 = self._add_bias_unit(a1, how='column')
        z2 = w2.dot(a1)
        a2 = self._sigmoid(z2)
        return a0, z1, a1, z2, a2

    def _L2_regularization(self, lambda_, w1, w2):
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + (np.sum(w2[:, 1:] ** 2)))

    def _L1_regularization(self, lambda_, w1, w2):
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_regularization(self.l1, w1, w2)
        L2_term = self._L2_regularization(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost
    
    def _get_gradient(self, a0, a1, a2, z1, y_enc, w1, w2):
        delta2 = a2 - y_enc
        z1 = self._add_bias_unit(z1, how='row')
        delta1 = w2.T.dot(delta2) * self._sigmoid_gradient(z1)
        delta1 = delta1[1:, :]
        grad1 = delta1.dot(a0)
        grad2 = delta2.dot(a1.T)


        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2

    def predict(self, X):
        if len(X.shape) != 2:
            raise AttributeError('Xは[n_samples, n_features]形式の配列')
        a0, z1, a1, z2, a2 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z2, axis = 0)
        return y_pred

    def fit(self, X, y, print_progress = False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('|rEpoch: {:d} / {;d}'.format(i + 1, self.epochs))
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                a0, z1, a1, z2, a2 = self._feedforward(X_data[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc = y_enc[:, idx], output = a2, w1 = self.w1, w2 = self.w2)
                self.cost_.append(cost)

                grad1, grad2 = self._get_gradient(a0 = a0, a1 = a1, a2 = a2, z1 = z1, y_enc = y_enc[:, idx], w1 = self.w1, w2 = self.w2)
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
            sys.stderr.write('|n')
            return self




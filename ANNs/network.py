import random
import numpy as np


class Network:

    SEED = 3

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        np.random.seed(self.SEED)
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x, save_activations=False):
        a = x
        if save_activations:
            activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            if save_activations:
                activations.append(a)
        return activations if save_activations else a

    def backprop(self, x, y):
        # feedforward, saving activations
        activations = self.feedforward(x, save_activations=True)
        h = activations[-1]
        # backpropagation
        del_h_wrt_z = h * (1 - h)
        del_E_wrt_z = self.cost_derivative(h, y) * del_h_wrt_z
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b[-1] = del_E_wrt_z
        nabla_w[-1] = np.outer(del_E_wrt_z, activations[-2])
        for l in range(2, self.num_layers):
            a = activations[-l]
            del_E_wrt_a = np.matmul(del_E_wrt_z[np.newaxis], self.weights[-l+1])[0]
            del_a_wrt_z = a * (1 - a)
            del_E_wrt_z = del_E_wrt_a * del_a_wrt_z
            nabla_b[-l] = del_E_wrt_z
            nabla_w[-l] = np.outer(del_E_wrt_z, activations[-l-1])
        return (nabla_b, nabla_w)

    def cost_derivative(self, h, y):
        return h - y

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(h == y) for h, y in test_results)

    def SGD(self, data_train, epochs, minibatch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n_train = len(data_train)
        for j in range(epochs):
            random.shuffle(data_train)
            for k in range(0, n_train, minibatch_size):
                minibatch = data_train[k:k+minibatch_size]
                self.minibatch_update(minibatch, eta)
            if test_data:
                print('Epoch {}: {} / {}.'.format(j+1, self.evaluate(test_data), n_test))
            else:
                print('Epoch {} complete.'.format(j))

    def minibatch_update(self, minibatch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in minibatch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(minibatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(minibatch))*nb for b, nb in zip(self.biases, nabla_b)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

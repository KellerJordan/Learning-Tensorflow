import random
import numpy as np

import sys
from pprint import pprint


## math functions ---------------------------------------------------------------------------------
def sigmoid(z):
    # return z * (z > 0) # ReLU
    return 1.0 / (1.0 + np.exp(-z)) # sigmoid

def sigmoid_prime(a):
    # return 1 * (a > 0) # ReLU
    return a * (1 - a) # sigmoid

## cost functions ---------------------------------------------------------------------------------
## return the derivative of C with respect to z^L
def QuadraticCost(h, y):
    return (h - y) * sigmoid_prime(h) # del_C_wrt_z^L
    # return h - y # del_C_wrt_h

def CrossEntropyCost(h, y):
    return h - y # del_C_wrt_z^L
    # return (h - y) / (h * (1 - h)) # del_C_wrt_h

## normalization functions ------------------------------------------------------------------------
def L2Normalizer(lmbda):
    return lambda w: lmbda * w

## artificial neural network ----------------------------------------------------------------------
class Network:

    SEED = 3

    def __init__(self, shape):
        self.num_layers = len(shape)
        np.random.seed(self.SEED)
        self.biases = [np.random.randn(y) for y in shape[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]

    # def large_weight_initializer(self):
    #     pass

    def feedforward(self, x, save_activations=False):
        a = x
        if save_activations:
            activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a) + b)
            if save_activations:
                activations.append(a)
        return activations if save_activations else a

    # uses denominator layout for matrix calculus
    def backpropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward, saving activations
        activations = self.feedforward(x, save_activations=True)
        h = activations[-1] # hypothesis
        # backpropagation
        del_C_wrt_z = self.cost_derivative(h, y)
        for l in range(1, self.num_layers):
            nabla_b[-l] = del_C_wrt_z
            nabla_w[-l] = np.outer(del_C_wrt_z, activations[-l-1])
            nabla_w[-l] += self.normalization_derivative(nabla_w[-l])
            if l < self.num_layers - 1:
                del_C_wrt_a = np.matmul(self.weights[-l].transpose(), del_C_wrt_z)
                a = activations[-l-1]
                del_C_wrt_z = del_C_wrt_a * sigmoid_prime(a)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(h == y) for h, y in test_results)

    def SGD(self, epochs, minibatch_size, eta,
            training_data, evaluation_data,
            cost=CrossEntropyCost,
            normalization=None,
            monitor_evaluation_accuracy=True):
        self.cost_derivative = cost
        self.normalization_derivative = normalization

        n_train = len(training_data)
        n_test = len(evaluation_data)

        from datetime import datetime
        time_start = datetime.now()
        for j in range(epochs):
            random.shuffle(training_data)
            for k in range(0, n_train, minibatch_size):
                minibatch = training_data[k:k+minibatch_size]
                self.minibatch_update(minibatch, eta)
            if monitor_evaluation_accuracy:
                print('Epoch {}: {} / {}.'
                      .format(j+1, self.evaluate(evaluation_data), n_test))
            else:
                print('Epoch {} complete.'.format(j))
        time_end = datetime.now()
        time_delta = str(time_end - time_start).rsplit('.', 1)[0]
        print(time_delta)

    def minibatch_update(self, minibatch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in minibatch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(minibatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(minibatch))*nb for b, nb in zip(self.biases, nabla_b)]

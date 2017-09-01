"""network.py
Network class and helpers written while reading the http://neuralnetworksanddeeplearning.com ebook.

Major differences between this and
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
are as follows:

- Normalization is treated the same as cost, rather than being unseparable from the code.
- backpropagate() uses a call to feedforward(), rather than its own implementation. Accordingly,
feedforward() contains the save_activations flag.
- backpropagate() uses a more linear algebra, and is more efficient generally in presentation and
computation.
- In cost functions and otherwise, the last layer of activations (a^L) is referred to as h,
for `hypothesis`, instead.
- Rather than dividing dC/dw by n when using to update weights, it is divided by m. I don't know
why the ebook divides by n rather than the minibatch size in the first place!
- SGD()'s execution duration is timed.

"""


import random
import numpy as np
import json

import sys


## math functions ---------------------------------------------------------------------------------
def sigmoid(z):
    clipped_z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped_z)) # sigmoid
    # return z * (z > 0) # ReLU

def sigmoid_prime(a):
    return a * (1 - a) # sigmoid
    # return 1 * (a > 0) # ReLU

## artificial neural network ----------------------------------------------------------------------
class Network:

    SEED = 3

    def __init__(self, shape, init, cost, norm=None):
        # network architecture, initialize weights/biases
        self.shape = shape
        self.num_layers = len(shape)
        np.random.seed(self.SEED)
        self.init = init
        self.biases, self.weights = self.init(shape)
        # network training
        self.cost = cost
        self.norm = norm

    def feedforward(self, x, save_activations=False):
        a = x
        if save_activations:
            activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
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
        del_C_wrt_z = self.cost.delta(h, y)
        for l in range(1, self.num_layers):
            nabla_b[-l] = del_C_wrt_z
            nabla_w[-l] = np.outer(del_C_wrt_z, activations[-l-1])
            # nabla_w[-l] += 0.0001 * np.sign(nabla_w[-l])
            if self.norm: nabla_w[-l] += self.norm.delta(self.weights[-l])
            if l < self.num_layers - 1:
                del_C_wrt_a = self.weights[-l].transpose() @ del_C_wrt_z
                a = activations[-l-1]
                del_C_wrt_z = del_C_wrt_a * sigmoid_prime(a)
        return (nabla_b, nabla_w)

    def minibatch_update(self, minibatch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # sum the derivatives of w with respect to cost over all training examples in batch
        for x, y in minibatch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update accordingly
        self.weights = [w - (eta/len(minibatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(minibatch))*nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, epochs, minibatch_size, eta_start,
            data_train, data_eval=None,
            early_stop=None,
            learning_rate_adjustment=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        n_train = len(data_train)
        if data_eval:
            n_eval = len(data_eval)

        from datetime import datetime
        time_start = datetime.now()

        eta = eta_start
        for j in range(epochs):
            random.shuffle(data_train)
            for k in range(0, n_train, minibatch_size):
                minibatch = data_train[k:k+minibatch_size]
                self.minibatch_update(minibatch, eta)
            print('Epoch {} complete.'.format(j+1))
            if monitor_evaluation_cost and data_eval:
                cost = self.total_cost(data_eval)
                evaluation_cost.append(cost)
                print('Cost on evaluation data: {}'.format(cost))
            if monitor_evaluation_accuracy and data_eval:
                accuracy = self.accuracy(data_eval)
                evaluation_accuracy.append(accuracy / n_eval)
                print('Accuracy on evaluation data: {} / {}'.format(accuracy, n_eval))
                if early_stop and early_stop.test(accuracy):
                    break
                if learning_rate_adjustment and learning_rate_adjustment.test(accuracy):
                    if eta < eta_start / 128:
                        break
                    print('Halving eta due to lack of improvement.')
                    eta *= 0.5
            if monitor_training_cost:
                cost = self.total_cost(data_train)
                training_cost.append(cost)
                print('Cost on training data: {}'.format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(data_train)
                training_accuracy.append(accuracy / n_train)
                print('Accuracy on training data: {} / {}'.format(accuracy, n_train))

        time_end = datetime.now()
        time_delta = str(time_end - time_start).rsplit('.', 1)[0]
        print(time_delta)

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    # measures of success
    def accuracy(self, data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        return sum(int(h == y) for h, y in test_results)

    def total_cost(self, data):
        test_results = [(self.feedforward(x), y) for x, y in data]
        loss = sum([self.cost.fn(h, y) for h, y in test_results])
        complexity = sum([self.norm.fn(w) for w in self.weights]) if self.norm else 0
        return (loss + complexity) / len(data)

    # could use numpy.save for parameters instead
    def save(self, filename):
        data = {'shape': self.shape,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'init': self.init.__name__,
                'cost': self.cost.__name__,
                'norm': self.norm.__name__}
        with open(filename, 'w') as f:
            json.dump(data, f)

def load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    import utils
    init = getattr(utils, data['init'])
    cost = getattr(utils, data['cost'])
    norm = getattr(utils, data['norm'])
    net = Network(data['shape'], init=init, cost=cost, norm=norm)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net

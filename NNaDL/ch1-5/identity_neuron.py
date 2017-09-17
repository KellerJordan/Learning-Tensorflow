"""identity_neuron.py
Uses gradient descent to compute best w_2, w_1, b such that
    x ~= w_2 * sigmoid(w_1 * x + b)
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class Network:
    def __init__(self):
        self.w2 = self.w1 = 0.5
        self.b = 0

    # quadratic
    def cost(self, h, y):
        return 0.5*np.sum((h - y)**2)

    def get_training_data(self, n):
        return np.random.uniform(size=n)
        # return np.arange(0, 1, 1.0/n)

    def SGD(self, eta, batch_size, per_epoch, epochs):
        best_C = 10000000000
        since_best = 0
        for j in range(epochs):
            C = 0
            for _ in range(per_epoch):
                x = self.get_training_data(batch_size)
                a = sigmoid(self.w1*x + self.b)
                h = self.w2*a
                C += self.cost(h, x)
                nabla_h = h - x
                nabla_z = nabla_h * self.w2*a*(1-a)
                self.w2 -= (eta/batch_size) * np.dot(nabla_h, a)
                self.w1 -= (eta/batch_size) * np.dot(nabla_z, x)
                self.b -= (eta/batch_size) * np.sum(nabla_z)

            if C < best_C:
                best_C = C
            else:
                since_best += 1
            if since_best >= 10:
                print('Halving eta')
                since_best = 0
                eta *= 0.5
            print('Epoch {} complete with C={}.'.format(j+1, C))

if __name__ == '__main__':
    net = Network()
    net.SGD(eta=0.3, batch_size=1000, per_epoch=1000, epochs=300)
    print(net.w2, net.w1, net.b)

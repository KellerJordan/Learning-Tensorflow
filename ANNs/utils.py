"""utils.py
Python file containing various utility classes for use in neural networks.

See network.py on more details about functions and difference from ebook.

"""


import numpy as np

## weight initializers ----------------------------------------------------------------------------
def NormalWeightInitializer(shape):
    biases = [np.random.randn(y) for y in shape[1:]]
    weights = [np.random.randn(y, x) / np.sqrt(x)
               for x, y in zip(shape[:-1], shape[1:])]
    return biases, weights

def LargeWeightInitializer(shape):
    biases = [np.random.randn(y) for y in shape[1:]]
    weights = [np.random.randn(y, x)
               for x, y in zip(shape[:-1], shape[1:])]
    return biases, weights

## cost functions ---------------------------------------------------------------------------------
## fn(): returns the value of the cost incurred
## delta(): returns the derivative of C with respect to z^L
class QuadraticCost:
    @staticmethod
    def fn(h, y):
        return 0.5 * np.linalg.norm(h - y)**2
    @staticmethod
    def delta(h, y):
        return (h - y) * sigmoid_prime(h)

class CrossEntropyCost:
    @staticmethod
    def fn(h, y):
        return np.sum(-np.nan_to_num(y * np.log(h) + (1 - y) * np.log(1 - h)))
    @staticmethod
    def delta(h, y):
        return h - y

## regularization functions -----------------------------------------------------------------------
## fn(): returns the cost incurred
## delta(): returns the value of C with respect to w
class L1Regularizer:
    def __init__(self, lmbda):
        self.lmbda = lmbda
    def fn(self, w):
        return self.lmbda * np.abs(w)
    def delta(self, w):
        return self.lmbda * np.sign(w)

class L2Regularizer:
    def __init__(self, lmbda):
        self.lmbda = lmbda
    def fn(self, w):
        return 0.5 * self.lmbda * np.linalg.norm(w)**2
    def delta(self, w):
        return self.lmbda * w

## early-stopping functions -----------------------------------------------------------------------
## test(): returns true if no improvement for last n epochs
class NoImprovementInN:
    def __init__(self, n):
        self.n = n
        self.max_accuracy = 0
        self.noimprove_count = 0
    def test(self, accuracy):
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.noimprove_count = 0
        else:
            self.noimprove_count += 1
        return self.noimprove_count >= self.n

# test(): return true if no improvement for last beta*(epochs so far) epochs
class CustomEarlyStop:
    def __init__(self, beta):
        self.beta = beta
        self.max_accuracy = 0
        self.noimprove_count = 0
        self.epoch_count = 0
    def test(self, accuracy):
        self.epoch_count += 1
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.noimprove_count = 0
        else:
            self.noimprove_count += 1
        return self.noimprove_count >= self.beta * self.epoch_count

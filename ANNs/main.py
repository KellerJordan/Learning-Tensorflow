import network
import numpy as np
from pprint import pprint

if __name__ == '__main__':

    # net = network.Network([5, 3, 2])
    # X = np.array([
    #     [1, 1, 1, 1, 1],
    #     [5, 5, 5, 5, 5],
    # ])
    # y = np.array([
    #     0,
    #     1,
    # ])
    # training_data = list(zip(X, y))
    # net.SGD(training_data, 30, 2, 3.0, test_data=training_data)

    net = network.Network([784, 30, 10])
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data()
    net.SGD(training_data, 15, 5, 3.0, test_data=test_data)

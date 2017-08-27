import network
import numpy as np
from pprint import pprint

if __name__ == '__main__':

    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data()
    net = network.Network([784, 100, 10])
    net.large_weight_initializer()
    net.SGD(
        training_data, 30, 5, .5,
        cost=network.CrossEntropyCost,
        normalization=network.L2Normalizer(lmbda=.1),
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True)

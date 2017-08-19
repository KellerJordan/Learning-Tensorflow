from network import Network
import numpy as np

X = np.array([
    [1, 1, 1, 1, 1],
    [5, 5, 5, 5, 5],
])

if __name__ == '__main__':
    # net = Network(784, [15, 10])
    net = Network([5, 3, 1])
    result = net.feedforward(X[0])
    print(result)

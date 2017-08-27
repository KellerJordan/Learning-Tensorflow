# python module to be run from command-line for artificial neural network experiments

def main():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data()
    import network
    net = network.Network([784, 100, 10])
    # net.large_weight_initializer()
    net.SGD(
        30, 5, .5,
        training_data,
        test_data,
        cost=network.CrossEntropyCost,
        normalization=network.L2Normalizer(lmbda=.1),
        monitor_evaluation_accuracy=True)


if __name__ == '__main__':
    main()

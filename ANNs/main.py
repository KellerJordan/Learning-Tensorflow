# python module to be run from command-line for artificial neural network experiments

def main():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data()
    import network
    net = network.Network([784, 30, 10],
                          cost=network.CrossEntropyCost,
                          norm=network.L2Regularizer(lmbda=5.0))
    _, evaluation_accuracy, _, _ = net.SGD(
        30, 10, .1,
        training_data, test_data,
        early_stopping=network.NoImprovementInN(10),
        monitor_evaluation_accuracy=True)

    from fig import plot
    plot(evaluation_accuracy)


if __name__ == '__main__':
    main()

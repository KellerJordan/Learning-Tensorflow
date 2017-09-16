# python module to be run from command-line for artificial neural network experiments

def main():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data()
    import network
    import utils
    net = network.Network([784, 30, 30, 30, 10],
                          init=utils.NormalWeightInitializer,
                          cost=utils.CrossEntropyCost,
                          norm=utils.L2Regularizer(lmbda=0.0001))
    _, evaluation_accuracy, _, _ = net.SGD(
        30, 10, .14,
        training_data, test_data,
        # early_stop=utils.NoImprovementInN(10),
        # learning_rate_adjustment=utils.NoImprovementInN(10),
        monitor_evaluation_accuracy=True)

    from fig import plot
    plot(evaluation_accuracy)


if __name__ == '__main__':
    main()

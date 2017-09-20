"""main.py

Main program to be run from the command line. Utilities provided by tfnetwork.py make neural
network experiments easy. Network architecture fully specified in arguments to Network().

"""

import argparse
import sys
import tempfile

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, DropoutLayer, LinearLayer
from network import softmax_cross_entropy_loss


    # train_step, accuracy = Network(
    #     layers=[ConvPoolLayer(32, [5, 5], [2, 2]),
    #             ConvPoolLayer(64, [5, 5], [2, 2]),
    #             FullyConnectedLayer(1024),
    #             DropoutLayer(keep_prob),
    #             LinearLayer(10)],
    #     loss=softmax_cross_entropy_loss,
    #     optimizer=tf.train.AdamOptimizer(1e-4))

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    input_dim, output_dim = 784, 10

    net = Network(
        input_dim, output_dim,
        layers=[ConvPoolLayer(20, [5, 5], [2, 2]),
                ConvPoolLayer(40, [5, 5], [2, 2]),
                FullyConnectedLayer(100, activation_fn=tf.nn.relu),
                DropoutLayer(0.5),
                LinearLayer(output_dim)],
        loss_func=softmax_cross_entropy_loss)

    epochs, batch_size = 60, 10

    net.train(mnist, epochs, batch_size,
              optimizer=tf.train.GradientDescentOptimizer(0.1/batch_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

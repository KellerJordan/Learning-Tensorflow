"""main.py

Main program to be run from the command line. Utilities provided by tfnetwork.py make neural
network experiments easy. Network architecture fully specified in arguments to Network().

"""

import argparse
import sys
import tempfile

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Network
from model import ConvPoolLayer, FullyConnectedLayer, DropoutLayer, LinearLayer
from model import softmax_cross_entropy_loss


FLAGS = None

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    input_dim, output_dim = 784, 10

    net = Network(
        input_dim, output_dim,
        layers=[ConvPoolLayer(32, [5, 5], [2, 2]),
                ConvPoolLayer(64, [5, 5], [2, 2]),
                FullyConnectedLayer(1024, activation_fn=tf.nn.relu),
                DropoutLayer(0.5),
                LinearLayer(output_dim)],
        loss_func=softmax_cross_entropy_loss)

    train_steps, batch_size = 20000, 50

    net.train(mnist, train_steps, batch_size,
              optimizer=tf.train.AdamOptimizer(1e-4),
              log_dir=FLAGS.log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tensorflow/mnist/2',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

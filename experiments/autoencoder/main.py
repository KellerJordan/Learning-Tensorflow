# main.py

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
    mnist_width = 28
    n_visible = mnist_width**2
    n_hidden = 500
    corruption_level = 0.3

    X = tf.placeholder(tf.float32, [None, n_visible])
    mask = tf.placeholder(tf.float32, [None, n_visible])

    W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
    W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                               minval=-W_init_max,
                               maxval=W_init_max)

    W = tf.Variable(W_init)
    b = tf.Variable(tf.zeros([n_hidden]))
    b_prime = tf.Variable(tf.zeros([n_visible]))

    # model
    Y = tf.nn.sigmoid(tf.matmul(mask * X, W) + b)
    Z = tf.nn.sigmoid(tf.matmul(Y, tf.transpose(W)) + b_prime)

    cost = tf.reduce_sum(tf.square(X - Z))
    train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    trX, trY = mnist.train.images, mnist.train.labels
    teX, teY = mnist.test.images, mnist.test.labels

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            for j in range(0, len(trX)-128, 128):
                input_ = trX[j:j+128]
                mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
                # mask_np = np.ones(input_.shape)
                sess.run(train_op, {X: input_, mask: mask_np})
            mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
            # mask_np = np.ones(teX.shape)
            print(i, sess.run(cost, {X: teX, mask: mask_np}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

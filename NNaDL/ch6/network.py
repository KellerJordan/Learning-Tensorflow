"""network.py

Tensorflow program to make neural network experiments easy. Interface from other programs is
designed to mimic the that of the Theano program described in Chapter 6 of the NNaDL eBook:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py
Other than that, an entirely different program.

"""

# pylint: disable=invalid-name

import tensorflow as tf
import numpy as np

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def softmax_cross_entropy_loss(y_, y_net):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_net)
    return tf.reduce_sum(cross_entropy)

def LinearLayer(n_out):
    def layer(nn_input):
        n_in = int(np.prod(nn_input.shape[1:]))
        if len(nn_input.shape) > 2:
            nn_input = tf.reshape(nn_input, [-1, n_in])
        W_fc = weight_variable([n_in, n_out])
        b_fc = bias_variable([n_out])
        return tf.matmul(nn_input, W_fc) + b_fc
    return layer

def FullyConnectedLayer(n_out, activation_fn=tf.nn.sigmoid):
    def layer(nn_input):
        linear_layer = LinearLayer(n_out)
        return activation_fn(linear_layer(nn_input))
    return layer

def ConvPoolLayer(out_channels, filter_shape, poolsize, activation_fn=tf.nn.relu):
    def layer(nn_image):
        # NWHC reshaping -- assumes is in NWH format already
        if len(nn_image.shape) < 4:
            nn_image = nn_image[..., None]
        in_channels = int(nn_image.shape[-1])
        W_conv = weight_variable(filter_shape+[in_channels, out_channels])
        b_conv = bias_variable([out_channels])
        activations_conv = tf.nn.conv2d(nn_image, W_conv, strides=[1, 1, 1, 1],
                                        padding='SAME', data_format='NHWC')
        h_conv = activation_fn(activations_conv + b_conv)
        pool_filter = [1]+poolsize+[1]
        h_pool = tf.nn.max_pool(h_conv, ksize=pool_filter, strides=pool_filter,
                                padding='SAME', data_format='NHWC')
        return h_pool
    return layer

# somewhat hacky solution to keeping nice syntax in main.py
KEEP_PROBS = []
def DropoutLayer(keep_prob):
    keep_var = tf.Variable(keep_prob)
    KEEP_PROBS.append(keep_var)
    def layer(nn_input):
        return tf.nn.dropout(nn_input, keep_var)
    return layer

class Network:

    def __init__(self,
                 input_dim, output_dim, layers,
                 loss_func):

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.y_ = tf.placeholder(tf.float32, shape=[None, output_dim])

        # layer pipeline
        side_len = int(np.sqrt(input_dim))
        activations = tf.reshape(self.x, [-1, side_len, side_len])
        for layer in layers:
            activations = layer(activations)
        y_net = activations

        self.loss = loss_func(self.y_, y_net)

        correct_prediction = tf.equal(tf.argmax(y_net, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self,
              dataset,
              epochs, batch_size,
              optimizer):

        accuracy = lambda data: self.accuracy.eval(feed_dict={
            self.x: data.images, self.y_: data.labels,
            **{keep_prob: 1.0 for keep_prob in KEEP_PROBS}})

        train_step = optimizer.minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                if i % 1 == 0:
                    print('Epoch {}, train accuracy {:.3%}'.format(i, accuracy(dataset.train)))
                for _ in range(dataset.train.num_examples // batch_size):
                    batch_x, batch_y = dataset.train.next_batch(batch_size)
                    train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y})
            print('Training complete, test accuracy {:.3%}'.format(accuracy(dataset.test)))

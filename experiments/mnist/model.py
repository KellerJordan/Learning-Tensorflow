"""network.py

Tensorflow program to make neural network experiments easy. Interface from other programs is
designed to mimic the that of the Theano program described in Chapter 6 of the NNaDL eBook:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py
Other than that, an entirely different program.

"""

# pylint: disable=invalid-name

import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial, name='W')
    tf.summary.histogram('weights', W)
    return W

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial, name='b')
    tf.summary.histogram('biases', b)
    return b

def softmax_cross_entropy_loss(y_, y_net):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_net)
    return tf.reduce_sum(cross_entropy)

def LinearLayer(n_out):
    def layer(nn_input):
        with tf.name_scope('linear'):
            n_in = int(np.prod(nn_input.shape[1:]))
            if len(nn_input.shape) > 2:
                nn_input = tf.reshape(nn_input, [-1, n_in])
            W_fc = weight_variable([n_in, n_out])
            b_fc = bias_variable([n_out])
            act = tf.matmul(nn_input, W_fc) + b_fc
            tf.summary.histogram('activations', act)
            return act
    return layer

def FullyConnectedLayer(n_out, activation_fn=tf.nn.sigmoid):
    def layer(nn_input):
        with tf.name_scope('fc'):
            linear_layer = LinearLayer(n_out)
            act = activation_fn(linear_layer(nn_input))
            tf.summary.histogram('activations', act)
            return act
    return layer

def ConvPoolLayer(out_channels, filter_shape, poolsize, activation_fn=tf.nn.relu):
    def layer(nn_image):
        with tf.name_scope('conv'):
            in_channels = int(nn_image.shape[-1])
            W_conv = weight_variable(filter_shape+[in_channels, out_channels])
            b_conv = bias_variable([out_channels])
            preactivations_conv = tf.nn.conv2d(
                nn_image, W_conv, strides=[1, 1, 1, 1], padding='SAME')
            act_conv = activation_fn(preactivations_conv + b_conv)
            tf.summary.histogram('act_conv', act_conv)
            pool_filter = [1]+poolsize+[1]
            h_pool = tf.nn.max_pool(
                act_conv, ksize=pool_filter, strides=pool_filter, padding='SAME')
            return h_pool
    return layer

# somewhat hacky solution to keeping nice syntax in main.py
KEEP_PROBS = []
def DropoutLayer(keep_prob):
    def layer(nn_input):
        with tf.name_scope('dropout'):
            keep_var = tf.Variable(keep_prob, name='keep_prob')
            KEEP_PROBS.append(keep_var)
            return tf.nn.dropout(nn_input, keep_var)
    return layer

class Network:

    def __init__(self,
                 input_dim, output_dim, layers,
                 loss_func):

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name='input')
        self.y_ = tf.placeholder(tf.float32, shape=[None, output_dim], name='label')

        with tf.name_scope('reshape'):
            side_len = int(np.sqrt(input_dim))
            x_image = tf.reshape(self.x, [-1, side_len, side_len, 1])
        tf.summary.image('input', x_image, 3)

        # layer pipeline
        activations = x_image
        for layer in layers:
            activations = layer(activations)
        y_net = activations

        with tf.name_scope('xent'):
            self.loss = loss_func(self.y_, y_net)
            tf.summary.scalar('cross_entropy', self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_net, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

    def train(self,
              dataset,
              train_steps, batch_size,
              optimizer,
              log_dir):

        with tf.name_scope('optimizer'):
            train_step = optimizer.minimize(self.loss)

        merged = tf.summary.merge_all()
        train_dir, test_dir = log_dir+'/train', log_dir+'/test'
        train_writer = tf.summary.FileWriter(train_dir)
        test_writer = tf.summary.FileWriter(test_dir)
        print('Saving summaries at %s and %s' % (train_dir, test_dir))
        train_writer.add_graph(tf.get_default_graph())

        def feed_dict(data):
            if data:
                data_x, data_y = data
                keep_dict = {}
            else:
                data_x, data_y = dataset.test.images, dataset.test.labels
                keep_dict = {keep_prob: 1.0 for keep_prob in KEEP_PROBS}
            return {self.x: data_x, self.y_: data_y, **keep_dict}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(train_steps):
                if i % 50 == 0: # log test accuracy
                    summary, acc = sess.run([merged, self.accuracy], feed_dict(False))
                    test_writer.add_summary(summary, i)
                    print('Accuracy at step {}: {:.3%}'.format(i, acc))
                batch = dataset.train.next_batch(batch_size)
                if i % 100 == 99: # record metadata
                    print('Recording metadata')
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step], feed_dict(batch),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    train_writer.add_summary(summary, i)
                else:
                    summary, _ = sess.run([merged, train_step], feed_dict(batch))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()

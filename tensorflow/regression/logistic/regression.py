"""regression.py

Tensorflow program to solve problem 1b) of Stanford 229 Problem Set #1.

"""

import tensorflow as tf
import numpy as np

def read_txt(f):
    str_arr = np.array([row.split() for row in f.read().split('\n')])
    return str_arr.astype(np.float)

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

theta = tf.Variable(np.zeros([2]), dtype=tf.float32)

z = tf.tensordot(x, theta, 1)[:, None]
log_l = tf.reduce_sum(-tf.log(tf.sigmoid(y_*z)))
delta_l = -y_*x * tf.sigmoid(-y_*z)
ddelta_l = x*x * tf.sigmoid(-y_*z) * tf.sigmoid(y_*z)

step = tf.reduce_sum(delta_l, axis=0) / tf.reduce_sum(ddelta_l, axis=0)
update_step = tf.assign_sub(theta, step)

with open('logistic_x.txt', 'r') as xfile:
    x_train = read_txt(xfile)

with open('logistic_y.txt', 'r') as yfile:
    y_train = read_txt(yfile)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x: x_train, y_: y_train}
    for _ in range(50):
        sess.run(update_step, feed_dict)

    print(sess.run(theta))

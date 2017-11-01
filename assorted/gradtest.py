import tensorflow as tf
import numpy as np

# model input
x = tf.placeholder(tf.float32, shape=[None, 5])
# model parameters
theta = tf.Variable(np.random.randn(5), dtype=tf.float32)
# any model with a second derivative
model = tf.sigmoid(x @ theta[:, None])

# tests
# The gradient of the gradient is NOT! the vector of second derivatives...
# It is the contraction of the hessian.
test1 = tf.gradients(tf.gradients(model, [theta])[0], [theta])[0]
test2 = tf.reduce_sum(tf.hessians(model, [theta])[0], axis=0)
# The only way to get the vector of second derivatives is to use the diagonal of the hessian.
test3 = tf.diag_part(tf.hessians(model, [theta])[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x: np.random.randn(100, 5)}
    for t in sess.run([test1, test2, test3], feed_dict):
        print(t)

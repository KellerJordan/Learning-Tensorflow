# Use gradient descent to get best parameters for an identity sigmoid neuron

import tensorflow as tf
import numpy as np

# model structure
x = tf.placeholder(tf.float32)
W1 = tf.Variable([.1], dtype=tf.float32)
W2 = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([0], dtype=tf.float32)
# single neuron model
h = W2 * tf.sigmoid(W1 * x + b)
# the best solution to below model cannot be learned using gradient descent (!!!)
# h = W2 * tf.sigmoid(W1 * x) + b
loss = tf.reduce_sum(tf.square(h - x))

# learning using gradient descent
alpha = tf.placeholder(tf.float32)
nabla_W2, nabla_W1, nabla_b = tf.gradients(loss, [W2, W1, b])
update_step = tf.group(
    tf.assign_sub(W2, alpha * nabla_W2),
    tf.assign_sub(W1, alpha * nabla_W1),
    tf.assign_sub(b, alpha * nabla_b))

# x_train = np.random.uniform(0, 1, [1000])
x_train = np.arange(0, 1, .01)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for _ in range(1000):
    # print(sess.run(loss, {x: x_train}))
    sess.run(update_step, {x: x_train, alpha: 0.001})
print(sess.run([W1[0], W2[0], b[0]]))

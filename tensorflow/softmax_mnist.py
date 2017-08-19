import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784]) # none to allow us to look at any number of examples at once
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

h = tf.matmul(x, W) + b
y = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)

def get_accuracy(): return sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

# A = []
# step_size = 20
num_iters = 1000
batch_size = 100
for i in range(num_iters):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    # if i % step_size == 0: A.append(get_accuracy())

print(get_accuracy())


# import matplotlib.pyplot as plt
# import numpy as np
# plt.plot(np.arange(0, num_iters, step_size), np.array(A))
# plt.show()

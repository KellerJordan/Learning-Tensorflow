import tensorflow as tf

# linear model
W = tf.Variable([.3])
b = tf.Variable([-.3])
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = 0.5 * tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# gradient descent hand-coded
if True:
    # model definition
    alpha = tf.placeholder(tf.float32)
    loss_delta = tf.gradients(loss, linear_model)[0] # or linear_model - y
    nabla_W = loss_delta * x
    nabla_b = loss_delta
    update_W = tf.assign(W, W - alpha*nabla_W)
    update_b = tf.assign(b, b - alpha*nabla_b)

    # running model
    sess = tf.Session()
    sess.run(init)
    for _ in range(100):
        # could use batches, not really enough training examples tho
        for i in range(len(x_train)):
            sess.run([update_W, update_b],
                     {x: x_train[i], y: y_train[i], alpha: 0.1})
    print(sess.run([W, b]))

# using tf.train API (as in https://www.tensorflow.org/get_started/get_started)
else:
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(init)
    for _ in range(1000):
        sess.run(train, {x: x_train, y: y_train})
    print(sess.run([W, b]))

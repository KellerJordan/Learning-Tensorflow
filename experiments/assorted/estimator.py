# Custom implementation of
# https://www.tensorflow.org/get_started/get_started#tfestimator

import tensorflow as tf
import numpy as np


use_LinearRegressor = False

if use_LinearRegressor:
    feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

else:
    def model_fn(features, labels, mode):
        W = tf.get_variable('W', [1], dtype=tf.float64)
        b = tf.get_variable('b', [1], dtype=tf.float64)
        # I wonder why features has to be a dictionary..
        h = W * features['x'] + b
        loss = tf.reduce_sum(tf.square(h - labels))
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=h,
            loss=loss,
            train_op=train)

    estimator = tf.estimator.Estimator(model_fn=model_fn)

# I don't understand why they have num_epochs=1000 for the train and eval functions.
# Doesn't seem to change anything since they are only run once on evaluation...
# Removing does nothing
# Can also switch (None, 1000) between num_epochs in input_fn and steps in estimator.train
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=4, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval}, y_eval, batch_size=4, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print(train_metrics)
print(eval_metrics)

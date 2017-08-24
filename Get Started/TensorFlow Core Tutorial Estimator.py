# tf.estimator simplifies:
# - running training loops
# - running evaluation loops
# - managing data sets

import os
import tensorflow as tf
# NumPy is used for loading, manipulating and preprocessing data
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Now we declare a feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Now we can create an estimator
# An estimator is a front-end for training and evaluation
# There are predefined types
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# We now create some data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8, 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# We have to create inputs for the function
# Number of epochs & batch size are defined
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can now train with 1000 steps
estimator.train(input_fn=input_fn, steps=1000)

# Now we can evaluate our model
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics : %r" % eval_metrics)

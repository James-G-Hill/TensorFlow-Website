# tf.estimator simplifies:
# - running training loops
# - running evaluation loops
# - managing data sets

import os
import tensorflow as tf
# NumPy is used for loading, manipulating and preprocessing data
import numpy as np

# Different integers here change the amount of logging reported
# when tensorflow is run
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Now we declare a feature
# This defines a numeric column called 'x'
# The shape gives the dimensions
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Now we can create an estimator
# An estimator is a front-end for training and evaluation
# There are predefined types
# 'feature_columns' here is both the parameter & variable name
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# We now create some data sets
# These are arrays created using the numpy library
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8, 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# We have to create inputs for the estimator
# The inputs take the form of functions
# Number of epochs & batch size are defined

# 'input_fn' uses the training arrays
# The number of iterations is given below during training
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
# 'train_input_fn' also uses the training arrays
# A number of epochs is given so that this can evaluated
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
# 'eval_input_fn' will be used to test the model on new data
# The evaluation data is passed in
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can now train with 1000 steps
# 'input_fun' below is both the parameter name & the variable name
# 'input_fun' from above is being passed into 'input_fn'
estimator.train(input_fn=input_fn, steps=1000)

# Now we can evaluate our model
# The training input is tested over 1000 epochs
# This will be accurate because it is identical to the input used to train
train_metrics = estimator.evaluate(input_fn=train_input_fn)
# The evaluation input is also tested over 1000 epochs
# This tests whether the ratios found in the trained input data
# are successfully applied to the new data
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics : %r" % eval_metrics)

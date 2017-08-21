import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters of the model
# Ideal paramaters are W = -1. and b = 1.
# These would result in the linear model producing y from x
# They are set deliberately here at wrong values
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input & output
# x is the number feeding the model
# W & b will be optimized to produce y from the calculation of W, x & b
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
# this is the difference between the linear model and y
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
# optimizer
# This adjusts the Variables in the model
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
# with W = -1 & b = 1, x_train passed to the module results in y_train
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

# initializes with W & b
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

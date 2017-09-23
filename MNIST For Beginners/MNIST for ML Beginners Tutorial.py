# Download the data from the tensorflow tutorials site
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# CREATE THE MODEL

# Placeholder to input MNIST images ('None' means 'any length')
# '784' is the number of pixels / dimensions
x = tf.placeholder(tf.float32, [None, 784])

# Variables are tensors that can be used & modified by the computation
# Variables are generally the model parameters
# In this case the initial values are all zeros
# They could have been other random numbers because these
# variables are going to be learnt by the computation
# '10' is the number of possible results (numerals)
# 'W' is for weight
W = tf.Variable(tf.zeros([784, 10]))
# 'b' is for bias
# 'b' has shape [10] so that it cana be added to the output
b = tf.Variable(tf.zeros([10]))

# matmul multiplies x & W before b is added
# softmax is applied to the result
# 'y' is our result
y = tf.nn.softmax(tf.matmul(x, W) + b)


# DEFINE LOSS & OPTIMIZER

# In this example we will use cross-entropy function to test our model
y_ = tf.placeholder(tf.float32, [None, 10])

# We implement the cross entropy function
# The logarithm of each y is calculated
# Then y_ is multiplied by the log of y
# Then the elements in 2nd dimension of y are summed together
# Then the mean is computed
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Gradient Descent Algorithm with learning rate of 0.5 is used
# Gradient Descent shifts each variable a little in the direction reducing cost
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# USE THE MODEL

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1000):
    # Randomly select batches of 100
    # Using small batches is called 'stochastic training'
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test
# tf.argmax gives the index of the highest entry in a tensor along an axis
# y is the prediction & y_ is the true result
# tf.equal returns a list of booleans
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# To calculate the fraction correct we cast the booleans to floats &
# then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Finally we output the results
print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))

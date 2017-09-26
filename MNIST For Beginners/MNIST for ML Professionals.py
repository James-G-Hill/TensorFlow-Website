from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

###
# Variables

no_of_training_iterations = 20000

###

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf. placeholder(tf.float32, [None, 10])


# Weights are initialized with noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Bias is slightly positive to avoid 'dead neurons'
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# This function creates a convolution layer
# The stride controls the movement of filters around the image (1 is small movement)
# Padding doesn't change the image in this case
def conv2d(x, W):
    return tf.nn.conv2d(
        x,
        W,
        strides=[1, 1, 1, 1],
        padding='SAME')


# This function creates a pooling layer
# The pooling is 'max pooling' which takes the max result from each partition
# The pooling layer progressively reduces the spatial size of the reprsentation
# This layer also reduces the number of parameters & amount of computation
# This controls overfitting
# 'ksize' is the size of the filter
# strides sets the movement of the filter
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')


# 1st Convolutional Layer

# The weight computes 32 features for 5x5 patches
# The 3rd parameter is the count of input channels
# The bias matches the output channel size for features
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# x (images) are reshaped to 4d tensors
# 2nd & 3rd dimensions correspond to image width & height
# 4th dimension corresponds to the number of colour channels
x_image = tf.reshape(x, [-1, 28, 28, 1])

# The image is convolved with the weight
# Then added with the bias
# Then reduced with the max pool to a 14x14 image
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 2nd Convolutional Layer

# The 2nd layer will have 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# The image size is now reduced to 7x7
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# The tensor from the pooling layer is reshaped, multiplied & biased
# An ReLU (Rectified Linear Units) layer is then applied
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout is then applied to reduce overfitting
# Dropout does this by randomly dropping units from the NN during training
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Finally we add a readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Training & Evaluation

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(no_of_training_iterations):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

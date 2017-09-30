import math
import tensorflow as tf
import os

""" Build an MNIST NN

Uses the 3-stage inference/loss/training pattern for model building

1. Inference: build the model for running a network
2. Loss: Adds layers to generate the loss
3. Training: Adds operations for generating & applying gradients

This file is not run but is used by the 'fully-connected' file'

"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
NUM_CLASSES = 10  # These are the outputs for digits 0 to 9
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# Graph Building

# This builds the graph that returns a tensor containing output predictions
def inference(images, hidden1_units, hidden2_units):

    # Hidden layer 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=0.1 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([hidden1_units]),
            name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden Layer 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits


# This builds the graph by adding the loss ops
def loss(logits, labels):

    # Convert placeholders to 64-bit ints
    labels = tf.to_int64(labels)

    # Add op to produce 1-hot labels
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='xentropy')

    # Average the cross entropy values across the batch dimension
    loss = tf.reduce_mean(
        cross_entropy,
        name='xentropy_mean')

    return loss


# Adds operations needed to minimize the loss
def training(loss, learning_rate):

    tf.summary.scalar('loss', loss)

    # The optimizer applies the gradients
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # The 'train_op' induces a full step of training
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

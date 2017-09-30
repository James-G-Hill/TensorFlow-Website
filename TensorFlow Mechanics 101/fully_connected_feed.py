import argparse
import os
import tensorflow as tf
import sys
import time
import graph_builder

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from six.moves import xrange


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Basic model parameters
FLAGS = None


# This function creates placeholder ops that define the inputs shapes
# These are inputs used by the rest of the model building code
def placeholder_inputs(batch_size):

    # Creates an array of floats with size 'batch_size X no. of pixels'
    images_placeholder = tf.placeholder(
        tf.float32,
        shape=(
            batch_size,
            graph_builder.IMAGE_PIXELS))

    # Creates an array of integers the same size as the batch
    labels_placeholder = tf.placeholder(
        tf.int32,
        shape=(batch_size))

    return images_placeholder, labels_placeholder


# Fills the feed dictionary for training a step
def fill_feed_dict(
        data_set,
        images_pl,
        labels_pl):

    # Returns separate images & labels from the data set
    images_feed, labels_feed, = data_set.next_batch(
        FLAGS.batch_size,
        FLAGS.fake_data)

    # Creates a dictionary from the images & labels
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }

    return feed_dict


# Runs one evaluation against the full epoch of data
def do_eval(
        sess,
        eval_correct,  # the tensor that returns no. of correct predictions
        images_placeholder,
        labels_placeholder,
        data_set):

    # true_count accumulates all predictions 'in_top_k' determines are correct
    true_count = 0  # Counts no. of correct predictions
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    # Evaluate the model on a given dataset
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(
            data_set,
            images_placeholder,
            labels_placeholder)
        true_count += sess.run(
            eval_correct,
            feed_dict=feed_dict)

    # Precision calculated by dividing true_count by number of examples
    precision = float(true_count) / num_examples

    print(' Num examples: %d  Num correct: %d  Precision @ 1 %0.04f' %
          (num_examples, true_count, precision))


# Run the training session
def run_training():

    # Get the setes of images & labels
    data_sets = input_data.read_data_sets(
        FLAGS.input_data_dir,
        FLAGS.fake_data)

    # The model will be built with the default graph
    with tf.Graph().as_default():

        # Placeholders created for images & labels
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        logits = graph_builder.inference(
            images_placeholder,
            FLAGS.hidden1,
            FLAGS.hidden2)

        loss = graph_builder.loss(logits, labels_placeholder)

        train_op = graph_builder.training(loss, FLAGS.learning_rate)

        # Adds an operation to compare logits to labels
        # Evaluation returns a function that automaticalaly scores each model
        #  output as correct if the true label can be found in k most-likely
        #  predictions
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Summaries are collected into a single Tensor during graph building
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        # Saver allows checkpoint files useful for restoring models
        saver = tf.train.Saver()

        sess = tf.Session()

        # FileWriter is instatntiated to write event files
        summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir,
            sess.graph)

        sess.run(init)

        # Training loop
        for step in xrange(FLAGS.max_steps):

            start_time = time.time()

            # Fill feed_dict with actual images & labels
            feed_dict = fill_feed_dict(
                data_sets.train,
                images_placeholder,
                labels_placeholder)

            # Rune 1 step of the model
            _, loss_value = sess.run(
                [train_op, loss],
                feed_dict=feed_dict)

            duration = time.time() - start_time

            # Print status every 100 steps
            if step % 100 == 0:

                # Print to standered output
                print('Step %d: loss = %.2f (%.3f sec)' % (
                    step,
                    loss_value,
                    duration))

                # Create a summary
                summary_str = sess.run(
                    summary,
                    feed_dict=feed_dict)

                # Add to the writer
                summary_writer.add_summary(
                    summary_str,
                    step)

                summary_writer.flush()

            # Save a checkpoint & evaluate the model every 1k steps
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:

                # Stores a file with current values of all trainable variables
                checkpoint_file = os.path.join(
                    FLAGS.log_dir,
                    'model.ckpt')

                saver.save(
                    sess,
                    checkpoint_file,
                    global_step=step)

                # Evaluate against the training set
                print('Training Data Eval:')

                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train)

                # Evaluate against the validation set
                print('Validation Data Eval:')

                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)

                print('Test Data Eval:')

                do_eval(
                    sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)


# Runs the entire program
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Learning rate
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    # Maximum steps
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,  # should be 2000
        help='Number of steps to run trainer.'
    )

    # Hidden 1
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )

    # Hidden 2
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )

    # Batch size
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )

    # Input data directory
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(
            os.getenv('TEST_TMPDIR', '/tmp'),
            'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )

    # Log directory
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(
            os.getenv('TEST_TMPDIR', '/tmp'),
            'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    # Fake data
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

FLAGS, unparsed = parser.parse_known_args()
tf.app.run(
    main=main,
    argv=[sys.argv[0]] + unparsed)

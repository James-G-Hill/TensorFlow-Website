# Import __future__ refers to a pseudo-module which programmers can use to
# enable new language features that are incompatible with the current
# interpreter
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main:

    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'w') as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, 'w') as f:
            f.write(raw)

    # Load datasets
    # Notice that the target column & features columns are typed separately
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Instantiate a deep neural network classifier

    # Specify all features have real-value data
    # Shape is for the 4 different features
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    
    # The classifier is created as a deep neural network
    # The NN has 3 layers of 10, 20 and 10 neurons.
    # The NN outputs 3 different results, relating to the 3 species of flowers
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir="/tmp/iris_model")

    

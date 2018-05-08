#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import dataset

tf.logging.set_verbosity(tf.logging.INFO)

TYPICAL = 0
NOVEL = 1


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = features["x"]
  input_layer = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), input_layer)
  input_layer = tf.map_fn(lambda frame: tf.image.random_flip_left_right(frame), input_layer)
  #input_layer = tf.map_fn(lambda frame: tf.image.random_flip_up_down(frame), input_layer)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 1]
  # Output Tensor Shape: [batch_size, 64, 64, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

  #bn1 = tf.layers.batch_normalization(inputs=conv1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64, 64, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

  #bn2 = tf.layers.batch_normalization(inputs=conv2, training=mode == tf.estimator.ModeKeys.TRAIN)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 16 * 16 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

  # Dense Layer
  # Densely connected layer with 512 neurons
  # Input Tensor Shape: [batch_size, 16 * 16 * 64]
  # Output Tensor Shape: [batch_size, 512]
  dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

  #bn3 = tf.layers.batch_normalization(inputs=dense, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 512]
  # Output Tensor Shape: [batch_size, 1]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  weights = tf.add(1, tf.multiply(labels, 328)) # will be weight = 1 for typical examples and 300 for novel
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)

  acc = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name="acc")

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1.0)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"], name="accuracy"),
      "true_positives": tf.metrics.true_positives(
          labels=labels, predictions=predictions["classes"], name="true_pos"),
      "true_negatives": tf.metrics.true_negatives(
          labels=labels, predictions=predictions["classes"], name="true_neg"),
      "false_positives": tf.metrics.false_positives(
          labels=labels, predictions=predictions["classes"], name="false_pos"),
      "false_negatives": tf.metrics.false_negatives(
          labels=labels, predictions=predictions["classes"], name="false_neg")
      }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load typical and novel datasets
  novel_results = '/home/hannah/src/MastcamCAE/results/DW_udr_12-8-3_7-5-3_nodrop_epochs15'
  typical_results = '/home/hannah/src/MastcamCAE/results/train_udr_12-8-3_7-5-3_nodrop_epochs15'

  typical_data = dataset.load_diff_images(typical_results) # Returns np.array
  typical_labels = np.zeros([typical_data.shape[0],1], dtype=np.int32)
  novel_data = dataset.load_diff_images(novel_results) # Returns np.array
  novel_labels = np.ones([novel_data.shape[0],1], dtype=np.int32)

  # Convert to training and eval sets
  train_data = np.concatenate([typical_data[:98700,:,:,:], novel_data[:300,:,:,:]])
  train_labels = np.concatenate([typical_labels[:98700], novel_labels[:300]])
  eval_data = np.concatenate([typical_data[98700:,:,:,:], novel_data[300:,:,:,:]])
  eval_labels = np.concatenate([typical_labels[98700:], novel_labels[300:]])

  # train_data = np.concatenate([typical_data[:4500,:,:,:], novel_data[:300,:,:,:]])
  # train_labels = np.concatenate([typical_labels[:4500], novel_labels[:300]])
  # eval_data = np.concatenate([typical_data[4500:,:,:,:], novel_data[300:,:,:,:]])
  # eval_labels = np.concatenate([typical_labels[4500:], novel_labels[300:]])

  # Create the Estimator
  multispec_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/home/hannah/src/MastcamCAE/saved_sessions/multispec_convnet_model_nodrop_eps15_udr_60k_seed42")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  debug_hook = tf_debug.TensorBoardDebugHook("ops5.sese.asu.edu:6064")

  # Train the model
  # train_input_fn = tf.estimator.inputs.numpy_input_fn(
  #     x={"x": train_data},
  #     y=train_labels,
  #     batch_size=50,
  #     num_epochs=None,
  #     shuffle=True)
  # multispec_classifier.train(
  #     input_fn=train_input_fn,
  #     steps=20000,
  #     hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = multispec_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
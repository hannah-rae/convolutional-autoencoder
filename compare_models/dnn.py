from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

from glob import glob

results_dw = glob('/home/hannah/src/MastcamCAE/results/DW_mse_udr_12-8-3_7-5-3_nodrop_epochs15/*')
results_train = glob('/home/hannah/src/MastcamCAE/results/train_mse_udr_12-8-3_7-5-3_nodrop_epochs15/*')

# Convert filenames to a list of integers (MSE)
results_dw = [[int(a.split('/')[-1].split('_')[0])] for a in results_dw]
results_train = [[int(a.split('/')[-1].split('_')[0])] for a in results_train]

# Create positive and negative labels
labels_dw = [1]*len(results_dw)
labels_train = [0]*len(results_train)

# Shuffle the lists with same seed as other models
np.random.seed(42)
np.random.shuffle(results_dw)
np.random.shuffle(results_train)

# Convert to training and eval sets
train_data = np.concatenate([results_train[:98700], results_dw[:300]])
train_labels = np.concatenate([labels_train[:98700], labels_dw[:300]])
weights = np.add(1, np.multiply(train_labels, 328))

eval_data = np.concatenate([results_train[98700:], results_dw[300:]])
eval_labels = np.concatenate([labels_train[98700:], labels_dw[300:]])

def main():

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
  weight_column = tf.feature_column.numeric_column('weight')

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[5, 10, 5],
                                          n_classes=2,
                                          weight_column=weight_column,
                                          model_dir="/tmp/dnn_mse")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data, "weight": weights},
      y=train_labels,
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=140000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data, "weight": np.ones([eval_data.shape[0]])},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  predictions = list(classifier.predict(input_fn=test_input_fn))
  predicted_classes = [p["classes"] for p in predictions]



  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


if __name__ == "__main__":
    main()
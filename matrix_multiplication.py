# Example for matrix / tensor multiplication
# This may represent a network with one layer with two neurons and four input neurons

import tensorflow as tf
import numpy as np

with tf.Session() as sess:

  # This is for logging the data flow graph:
  merged = tf.merge_all_summaries()
  writer=tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)
  
  # This defines the data flow graph:
  input_features = tf.constant(np.reshape([1, 0, 0, 1], (1, 4)).astype(np.float32))
  weights = tf.constant(np.random.randn(4, 2).astype(np.float32))
  output_layer = tf.matmul(input_features, weights)
  
  # This runs / evaluates the data flow graph and prints the result:
  print "Input:"
  print input_features.eval()
  print "Weights:"
  print weights.eval()
  print "Output:"
  print output_layer.eval()

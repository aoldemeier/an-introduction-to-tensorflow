# Example for a simple calculation with TensorFlow
# This may represent a simple feedforward network with one hidden layer
# Plot the function with WolframAlpha:
# plot tanh(4.0*tanh(-1.0*x-0.5*y+0.2)+2.0*tanh(-2.0*x+3.0*y+0.6)+4.2)

import tensorflow as tf
import numpy as np

# This defines the model (data flow graph):
x = tf.placeholder("float", shape=(1, 2))
w_layer1 = tf.constant([[-1.0, -0.5],[-2.0, 3.0]], shape=(2, 2))
b_layer1 = tf.constant([0.2, 0.6])
y_layer1 = tf.tanh(tf.matmul(x, w_layer1) + b_layer1)
w_layer2 = tf.constant([[4.0, 2.0]], shape=(2, 1))
b_layer2 = tf.constant([4.2])
y = tf.tanh(tf.matmul(y_layer1, w_layer2) + b_layer2)

with tf.Session() as sess:

  # This is for logging the data flow graph:
  writer = tf.summary.FileWriter("/tmp/tensorflowlogs", sess.graph)
  
  # This runs / evaluates the data flow graph and prints the result:
  random_x = np.random.randn(1, 2)
  print "x:"
  print x.eval(feed_dict={x:random_x})
  
  print "=> y:"
  print y.eval(feed_dict={x:random_x})

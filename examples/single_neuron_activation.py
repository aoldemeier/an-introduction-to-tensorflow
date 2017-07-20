# Example for a simple calculation with TensorFlow
# This may represent a single neuron with two inputs and a tanh activation function

import tensorflow as tf

# This defines the model (data flow graph):
x = tf.constant([1.2, -0.3], shape=(1, 2))
w = tf.constant([-0.2, 3.1], shape=(2, 1))
b = tf.constant(1.0)
y = tf.tanh(tf.matmul(x, w) + b)

with tf.Session() as sess:

  # This is for logging the data flow graph:
  writer = tf.summary.FileWriter("/tmp/tensorflowlogs", sess.graph)
  
  # This runs / evaluates the data flow graph and prints the result:
  print "x:"
  print x.eval()
  
  print "w:"
  print w.eval()
  print "b:"
  print b.eval()
  
  print "=> y:"
  print y.eval()

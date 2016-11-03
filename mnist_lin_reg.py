# This code is based on
# https://github.com/PacktPublishing/Getting-Started-with-TensorFlow/blob/master/Chapter%204/logistic_regression.py

# %matplotlib inline

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt




# Import MINST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)




#####################################
# Create model / compuational graph:
#####################################

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model (output function of the network)
actual_activation = tf.nn.softmax(tf.matmul(x, W) + b)

# Cost function: Euclidean distance between actual activation and ideal output
cost = tf.sqrt(tf.reduce_sum(tf.square(y - actual_activation)))

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost) 




#####################################
# Launch the graph:
#####################################
with tf.Session() as sess:
    
    # This is for logging the data flow graph:
    merged = tf.merge_all_summaries()
    writer=tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)
    
    sess.run(tf.initialize_all_variables())
    
    print ("Training...")
    
    for i in range(mnist.train.num_examples):
        batch_xs, batch_ys = mnist.train.next_batch(1)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

    print ("Training phase finished... Testing...")
    
    activations_max_indices = tf.argmax(actual_activation, 1).eval({x: mnist.test.images, y: mnist.test.labels})
    # Test model
    correct_prediction = tf.equal(tf.argmax(actual_activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print ("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

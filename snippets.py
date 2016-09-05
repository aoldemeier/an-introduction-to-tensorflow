# To start TensorBoard:
# docker exec container-name tensorboard --logdir=/tmp/tensorflowlogs

with tf.Session() as sess:
  merged = tf.merge_all_summaries()
  writer=tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)

# Example for matrix / tensor multiplication

import tensorflow as tf
import numpy as np

with tf.Session() as sess:

  # This is for logging the data flow graph:
  merged = tf.merge_all_summaries()
  writer=tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)
  
  # This defines the data flow graph:
  input_features = tf.constant(np.reshape([1, 0, 0, 1], (1, 4)).astype(np.float32))
  weights = tf.constant(np.random.randn(4, 2).astype(np.float32))
  output = tf.matmul(input_features, weights)
  
  # This runs / evaluates the data flow graph and prints the result:
  print "Input:"
  print input_features.eval()
  print "Weights:"
  print weights.eval()
  print "Output:"
  print output.eval()





# EXAMPLE FOR ONE LAYER NET

# This code is based on
# https://github.com/PacktPublishing/Getting-Started-with-TensorFlow/blob/master/Chapter%204/logistic_regression.py

%matplotlib inline

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
actual_activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Cost function: Euclidean distance between actual activation and 
cost = tf.sqrt(tf.reduce_sum(tf.square(y - actual_activation)))

# Optimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

#Plot settings
avg_set = []
epoch_set=[]

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:

    # This is for logging the data flow graph:
    merged = tf.merge_all_summaries()
    writer=tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)

    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = \
                      mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, \
                     feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, \
                                 feed_dict={x: batch_xs, \
                                            y: batch_ys})/total_batch
        
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print ("Training phase finished")
    
    activations_max_indices = tf.argmax(actual_activation, 1).eval({x: mnist.test.images, y: mnist.test.labels})
    # Test model
    correct_prediction = tf.equal(tf.argmax(actual_activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
plt.plot(epoch_set,avg_set, 'o', label='Training phase')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend()
plt.show()













# TESTING

%matplotlib inline

import numpy as np

index = 2

formatted_array = np.array(mnist.test.images[index]).reshape (28,28)
plt.imshow(formatted_array)
print("Correct digit: ", np.argmax(mnist.test.labels[index]))

print("Prediction: ", activations_max_indices[index])

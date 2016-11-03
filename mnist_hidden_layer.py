# %matplotlib inline

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


BATCH_SIZE = 10
EPOCHS = 100

# Import MINST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)




#####################################
# Create model / computational graph:
#####################################

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W1 = tf.Variable(tf.truncated_normal([784, 200]))
b1 = tf.Variable(tf.truncated_normal([200]))
W2 = tf.Variable(tf.truncated_normal([200, 10]))
b2 = tf.Variable(tf.truncated_normal([10]))

# Construct model (output function of the network)
actual_activation = tf.nn.softmax(
    tf.matmul(
        tf.sigmoid(tf.matmul(x, W1) + b1),
        W2) + b2
)

# Cost function: Euclidean distance between actual activation and ideal output
cost = tf.reduce_sum(tf.square(y - actual_activation))

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)




#####################################
# Launch the graph:
#####################################
with tf.Session() as sess:
    
    # This is for logging the data flow graph:
    merged = tf.merge_all_summaries()
    writer=tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)
    
    sess.run(tf.initialize_all_variables())
    
    print ("Training...")

    for epoch in range(EPOCHS):
        print("Epoch ", epoch)
        for i in range(mnist.train.num_examples / BATCH_SIZE):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        activations_max_indices = tf.argmax(actual_activation, 1).eval({x: mnist.test.images, y: mnist.test.labels})
        # Test model
        correct_prediction = tf.equal(tf.argmax(actual_activation, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print ("Cost ", cost.eval({x: mnist.test.images, y: mnist.test.labels}))

    print ("Training phase finished... Testing...")
    


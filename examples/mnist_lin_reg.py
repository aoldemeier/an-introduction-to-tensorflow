# This code is based on
# https://github.com/PacktPublishing/Getting-Started-with-TensorFlow/blob/master/Chapter%204/logistic_regression.py

%matplotlib inline

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data




# Import MINST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)




#####################################
# Create model / computational graph:
#####################################

# tf Graph Input
x = tf.placeholder("float", shape=(None, 784)) # mnist data image of shape 28*28=784
y = tf.placeholder("float", shape=(None, 10))  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))

# Construct model (output function of the network)
actual_activation = tf.nn.softmax(tf.matmul(x, W) + b)

# Cost function: Euclidean distance between actual activation and ideal output
cost = tf.sqrt(tf.reduce_sum(tf.square(y - actual_activation)))

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 




#####################################
# Launch the graph:
#####################################
with tf.Session() as sess:
    
    # This is for logging the data flow graph:
    writer = tf.summary.FileWriter("/tmp/tensorflowlogs", sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    print ("Training...")
    
    BATCH_SIZE=1
    for i in range(mnist.train.num_examples / BATCH_SIZE):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # Optimize weights
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

    print ("Training phase finished... Testing...")
    
    # Calculate actual predictions
    predictions = tf.argmax(actual_activation, 1)
    test_ground_truths = tf.argmax(y, 1)
    predictions_evaluated = predictions.eval(feed_dict={x: mnist.test.images, y:mnist.test.labels})
    
    # Compare with ground truth
    is_correct_prediction = tf.equal(predictions, test_ground_truths)
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, "float"))
    
    print ("Model accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

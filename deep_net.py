import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is a feed forward neural network as the data is going straight through
''''
take input with weights > hidden layer 1
hidden layer 1 > activation function
hidden layer 1 with weights > hidden layer 2
hidden layer 2 > activation function 
hidden layer 2 with weights > output layer 

compare the output with the intended output
using a cost or loss function 

(cross entropy) how close we are to the intended output
use an optimizer to attempt to minimize cost (adamoptimizer, gradient descent, adagrad)
goes backwards and manipulates the weights (back propagation)

feed forward + back prop = epoch (exactly one cycle, do it approx 20 times)
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# onehot means that one component is hot and the rest are off
# useful for multi class (0 - 9 , ten output classes)
"""
for onehot, an output of 0 would be [1,0,0,0,0,0,0,0,0,0,0]
only one of the parameters in the tensor is active
"""
# these will be the hidden layers, with each having 500 nodes,
# they do not have to be identical, they can change and be unique values
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

# goes through 100 batches of features and feeds them through our network at a time
# manipulates the weights
# goes through 100 images at a time
batch_size = 100

# Picture comes in as a 28 by 28, but we flatten it
# if something not of the shape specified is inputted, there will be an error
# x is the data being fed and y is the label of that data
x = tf.placeholder(tf.float32, [None, 784])  # 784 pixels wide, matrix is height by width,
#  the height is none, width is 28 x 28, squash to become a 784 vector
y = tf.placeholder(tf.float32)


def neural_network_model(data):
    # a tensorflow variable, the weights
    # specify the shape
    # creates a tensor of your data using random numbers
    # the shape of the tensor should be 2 dimensional and should actually be referred to as a matrix
    # input * weights + bias

    # bias are added at the end so they do not need a shape
    # biases are useful if the input is 0, which would make no neurons fire

    # multiply the 784 by the number of nodes in the first hidden layer
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    # keep i mind that the 784 is the input, but changes to the nodes for the hidden layers later on

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    # the number of biases here is presented as the one-hot
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # this is the sum box, multiply the data value by the l1 weight and add the l1 bias
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # relu is the threshold, an activation function:  f(x)= x^{+}= max(0,x)
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


# x is the input data
def train_neural_network(x):
    prediction = neural_network_model(x)
    # this calculates the difference between the prediction and the y, being the label
    # the cost is the difference, you want a small cost value
    # will return a 1-d tensor of length batch with the same type as logits
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # AdamOptimizer has a default parameter, being a learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # feed forward and back propagation
    hm_epochs = 10

    # session has begun running
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # underscore is short for variable we do not care about
            # we have the total amount of samples and a batch size
            # we then know how many times we need to cycle
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # chunk through the data set for you, this is prebuilt to do it for you
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                # epoch loss is actually the loss for each epoch
                epoch_loss += c
            print('epoch', epoch, 'completed out of ', hm_epochs, 'loss', epoch_loss)

        # anything from the for loop up until now is just training the network

        # argmax will return the index of the maximum value
        # see if the index values are the same, check if the one-hots are identical
        # compare the prediction with the actual label
        correct = tf.equal(tf.arg_max(prediction, 1), tf.argmax(y, 1))

        # cast will change the correct into a float
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # compare the images with the labels
        print('accuracy', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)


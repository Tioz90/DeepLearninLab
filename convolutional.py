import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''Initialise the weights to small positive values as we are using ReLU'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

'''Initialise the bias'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''Convolution with stride 1 [batch, height, width, channels] in each direction and padding so size of
input is same as output'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

'''Max pooling layer with window size 2x2'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''Define MLP meodel'''
def MLP(x):
    '''Define variables'''
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    '''Define the model'''
    y = tf.matmul(x, W) + b

    return y

'''Define CNN model'''
def CNN(x):
    '''compute 32 features using a 5x5 patch'''
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    '''Bring image x in line with the weights'''
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    '''Convolve x and pass into the max pooling layer'''
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    '''Define 64 features of size 5x5'''
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    '''Second layer'''
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    '''Define the weights for 1024 neurons'''
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    '''Reshape the 7x7 output to feed into 1024 fully connected neurons'''
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    '''Readout layer'''
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y

'''Function to train and print models'''
def run(model, mnist, validation, batch_size, n_epochs, learning_rate):

    '''Start session'''
    sess = tf.InteractiveSession()

    '''Declare input/output
       NB: the shape parameter is optional but helps with debugging'''
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    '''Define the model'''
    y = model(x)

    '''Define loss function'''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    '''Define optimiser'''
    trainer = tf.train.AdamOptimizer(learning_rate)
    train_step = trainer.minimize(cross_entropy)

    '''Check performance'''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    '''Arrays to log values'''
    accuracyTrain = []
    accuracyValidation = []

    num_batches = n_epochs * (mnist.train.num_examples // batch_size)
    for i in range(num_batches):
        X_batch, Y_batch = mnist.train.next_batch(batch_size)

        if i % 100 == 0:
            accuracyTrain.append(accuracy.eval(feed_dict={x: X_batch, y_: Y_batch}))
            accuracyValidation.append(accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))

        sess.run(train_step, {x: X_batch, y_: Y_batch})

    '''Plot accuracy'''
    print(model.__name__, "accuracy after", i, "batches:", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    plt.plot(accuracyTrain, label="Training")
    plt.plot(accuracyValidation, label="Validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Batches")
    plt.title(model.__name__+" accuracy")
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0.7, 1.0])
    plt.show()

    tf.reset_default_graph()
    sess.close()

# -------------------------------------------------------------------------------------#

def main():

    learning_rate = 1e-2
    batch_size = 64
    n_epochs = 1

    '''Import MNIST data'''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    val_size = mnist.validation.num_examples

    run(MLP, mnist, val_size, batch_size, n_epochs, learning_rate)
    run(CNN, mnist, val_size, batch_size, n_epochs, learning_rate)

if __name__ == "__main__":
    main()
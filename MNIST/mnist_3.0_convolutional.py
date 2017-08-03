import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

mnist = mnist_data.read_data_sets('data', one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)

K = 4
L = 8
M = 12
N = 200

def conv2d(X_input, filter_size, stride, in_channels, out_channels):
    W = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_channels]))
    Y = tf.nn.relu(tf.nn.conv2d(X_input, W, strides=[1, stride, stride, 1], padding='SAME') + b)
    return Y

def fully_connected_layer(X_input, in_dim, out_dim):
    W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_dim]))
    Y = tf.nn.relu(tf.matmul(X_input, W) + b)
    return Y

def softmax_layer(X_input, in_dim, out_dim):
    W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_dim]))
    logits = tf.matmul(X_input, W) + b
    Y = tf.nn.softmax(logits)
    return logits, Y

stride1 = 1
stride2 = 2

Y1 = conv2d(X, 5, stride1, 1, K)
Y2 = conv2d(Y1, 4, stride2, K, L)
Y3 = conv2d(Y2, 4, stride2, L, M)

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = fully_connected_layer(YY, 7 * 7 * M, N)
Ylogits, Y = softmax_layer(Y4, N, 10)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100.

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_X, batch_Y = mnist.train.next_batch(100)

    max_learning_rate = .003
    min_learning_rate = .0001
    decay_speed = 2000.0
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    train_data = {X: batch_X, Y_: batch_Y, learning_rate: lr}

    sess.run(train_step, feed_dict=train_data)

    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('Iteration {}:'.format(i))
        print('Train: {} {}'.format(a, c))
        test_data = {X: mnist.test.images, Y_: mnist.test.labels, learning_rate: lr}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('Test: {} {}'.format(a, c))

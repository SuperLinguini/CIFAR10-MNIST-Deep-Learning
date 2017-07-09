import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

mnist = mnist_data.read_data_sets('data', one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

def fully_connected_layer(input, size_in, size_out):
    W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    layer = tf.nn.sigmoid(tf.matmul(input, W) + b)
    return layer

def fully_connected_layer_relu_dropout(input, size_in, size_out):
    W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    layer = tf.nn.dropout(tf.nn.relu(tf.matmul(input, W) + b), pkeep)
    return layer

def softmax_layer(input, size_in, size_out):
    W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    logits = tf.matmul(input, W) + b
    layer = tf.nn.softmax(logits)
    return logits, layer

K = 784
L = 200
M = 100
N = 60
O = 30
P = 10

XX = tf.reshape(X, [-1, K])
Y1 = fully_connected_layer_relu_dropout(XX, K, L)
Y2 = fully_connected_layer_relu_dropout(Y1, L, M)
Y3 = fully_connected_layer_relu_dropout(Y2, M, N)
Y4 = fully_connected_layer_relu_dropout(Y3, N, O)
Ylogits, Y = softmax_layer(Y4, O, P)

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

    train_data = {X: batch_X, Y_: batch_Y, learning_rate: lr, pkeep: .8}

    sess.run(train_step, feed_dict=train_data)

    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('Iteration {}:'.format(i))
        print('Train: {} {}'.format(a, c))
        test_data = {X: mnist.test.images, Y_: mnist.test.labels, learning_rate: lr, pkeep: .8}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('Test: {} {}'.format(a, c))

import tensorflow as tf
import math
from numpy.random import choice

from CIFAR10 import cifar10

tf.set_random_seed(0)

X = tf.placeholder(tf.float32, [None, cifar10.img_size, cifar10.img_size, cifar10.num_channels])
Y_ = tf.placeholder(tf.float32, [None, cifar10.num_output])
learning_rate = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

K = 6
L = 12
M = 24
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

Y1 = conv2d(X, 6, stride1, cifar10.num_channels, K)
Y2 = conv2d(Y1, 5, stride2, K, L)
Y3 = conv2d(Y2, 4, stride2, L, M)

YY = tf.reshape(Y3, shape=[-1, 8 * 8 * M])

Y4 = fully_connected_layer(YY, 8 * 8 * M, N)
YY4 = tf.nn.dropout(Y4, keep_prob=pkeep)
Ylogits, Y = softmax_layer(Y4, N, 10)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100.

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(init)

cifar10.load_train_test_data()

for i in range(100000):
    rand_ints = choice(50000, 100)
    batch_X = cifar10.X_train[rand_ints, :, :, :]
    batch_Y = cifar10.y_train[rand_ints, :]
    # batch_X, batch_Y = cifar10.get_batch(100)

    max_learning_rate = .003
    min_learning_rate = .0001
    decay_speed = 2000.0
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # print(type(batch_X), type(batch_Y))
    # print(batch_X.shape, batch_Y.shape)
    # print(i)

    # batch_X = batch_X.eval(session=sess)
    # batch_Y = batch_Y.eval(session=sess)

    # coord = tf.train.Coordinator()
    # # wake up the threads
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    # batch_X, batch_Y = sess.run([batch_X, batch_Y])
    #
    # # When done, ask the threads to stop.
    # coord.request_stop()
    # # Wait for threads to finish.
    # coord.join(threads)


    train_data = {X: batch_X, Y_: batch_Y, learning_rate: lr, pkeep: 0.75}

    sess.run(train_step, feed_dict=train_data)

    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('Iteration {}:'.format(i))
        print('Train: {} {}'.format(a, c))
        images = cifar10.X_test
        labels = cifar10.y_test
        test_data = {X: images, Y_: labels, learning_rate: lr, pkeep: 1.0}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('Test: {} {}'.format(a, c))

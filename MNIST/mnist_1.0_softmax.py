import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print('TF ' + tf.__version__)
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets('data', one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 784])

Y = tf.nn.softmax(tf.matmul(XX, W) + b)

cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(.005).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    sess.run(train_step, feed_dict=train_data)

    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print('Iteration {}:'.format(i))
        print('Train: {} {}'.format(a, c))
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print('Test: {} {}'.format(a, c))




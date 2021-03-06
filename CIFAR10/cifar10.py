import pickle
import numpy as np
import tensorflow as tf

data_dir = './cifar-10-batches-py/'
num_channels = 3
img_size = 32
num_output = 10

X_train = None
y_train = None
X_test = None
y_test = None

def read_pickle(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict


def convert_vectors_to_images(data):
    # Convert uint8 data to float32
    data_float = np.array(data, dtype=np.float32) / 255.

    # Convert 10000x3072 rank 2 tensor to 10000x32x32x3 rank 4 tensor
    formatted_data = data_float.reshape([-1, num_channels, img_size, img_size])
    formatted_data = formatted_data.transpose([0, 2, 3, 1])

    return formatted_data


def one_hot_encode_labels(labels):
    if isinstance(labels, list):
        labels = np.array(labels)
    one_hot = np.zeros([labels.size, num_output])
    one_hot[np.arange(labels.size), labels.astype(np.uint8)] = 1
    return one_hot


def get_label_names():
    data_dict = read_pickle(data_dir + 'batches.meta')
    return data_dict[b'label_names']


def get_training_data():
    X_train = np.array([], dtype=np.float32).reshape(0, img_size, img_size, num_channels)
    y_train = np.array([], dtype=np.float32)

    for i in range(1,6):
        data_dict = read_pickle(data_dir + 'data_batch_{}'.format(i))

        instances = data_dict[b'data']
        instances = convert_vectors_to_images(instances)

        labels = data_dict[b'labels']

        print(X_train.shape, instances.shape, y_train.shape)

        X_train = np.vstack([X_train, instances])
        y_train = np.append(y_train, labels)

    y_train = one_hot_encode_labels(y_train)

    return X_train, y_train


def get_test_data():
    data_dict = read_pickle(data_dir + 'test_batch')

    X_test = data_dict[b'data']
    X_test = convert_vectors_to_images(X_test)

    y_test = data_dict[b'labels']
    y_test = one_hot_encode_labels(y_test)

    return X_test, y_test

def load_train_test_data():
    global X_train, y_train, X_test, y_test
    X_train, y_train = get_training_data()
    X_test, y_test = get_test_data()

def get_batch(batch_size=100):
    X = tf.stack(X_train)
    y = tf.stack(y_train)
    image_batch, label_batch = tf.train.shuffle_batch([X, y], num_threads=4, capacity=50000, min_after_dequeue=10000,
                                                      batch_size=batch_size, enqueue_many=True)
    return image_batch, label_batch
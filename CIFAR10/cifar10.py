import tensorflow as tf
import pickle
import numpy as np

data_dir = './cifar-10-batches-py/'
num_channels = 3
img_size = 32

def read_pickle(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict

def convert_vectors_to_images(data):
    # Convert uint8 data to float32
    data_float = np.array(data, dtype=np.float32) / 255.

    # Convert 10000x3072 rank 2 tensor to 10000x32x32x3 rank 4 tensor
    formatted_data = data.reshape([-1, num_channels, img_size, img_size])
    formatted_data = formatted_data.transpose([0, 2, 3, 1])

    return formatted_data

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

    return X_train, y_train

def get_test_data():
    data_dict = read_pickle(data_dir + 'test_batch')

    X_test = data_dict[b'data']
    X_test = convert_vectors_to_images(X_test)

    y_test = data_dict[b'labels']

    return X_test, y_test

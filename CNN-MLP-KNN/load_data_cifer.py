import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    x_train = []
    y_train = []
    for batch in range(1, 6):
        file_path = os.path.join(data_dir, f"data_batch_{batch}")
        batch_data = unpickle(file_path)
        x_train.append(batch_data[b'data'])
        y_train.extend(batch_data[b'labels'])
    x_train = np.vstack(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train)
    test_data = unpickle(os.path.join(data_dir, "test_batch"))
    x_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_data[b'labels'])
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

# Τεστ μορφής δεδομένων
# data_dir = 'cifar-10-batches-py'
# (x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)
# print("Training data shape:", x_train.shape, y_train.shape)
# print("Test data shape:", x_test.shape, y_test.shape)

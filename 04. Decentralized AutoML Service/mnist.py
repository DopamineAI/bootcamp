from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
import numpy as np
import math
import matplotlib.pyplot as plt

__author__ = 'uyerushalmi'


class MnistDataFiltered(object):
    def __init__(self, train_from_idx, train_to_idx, included_characters):
        # input image dimensions
        img_rows, img_cols = 28, 28
        self.num_classes = 10

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train[train_from_idx:train_to_idx, :, :, :]
        y_train = y_train[train_from_idx:train_to_idx]

        if not included_characters is None:
            train_idx = np.isin(y_train, included_characters)
            test_idx = np.isin(y_test, included_characters)
            y_train = y_train[train_idx]
            x_train = x_train[train_idx]
            y_test = y_test[test_idx]
            x_test = x_test[test_idx]
  
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train/255
        self.x_test = x_test/255

        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)


def plot_mnist(mnist, n, ncols):
    nrows = math.ceil(n/ncols + 0.5)
    fig = plt.figure()
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mnist[i].reshape([28,28]), cmap='gray')
    plt.show()

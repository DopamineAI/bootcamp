from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K

__author__ = 'uyerushalmi'


class MnistData(object):
    def __init__(self, train_from_idx, train_to_idx):
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

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train/255
        self.x_test = x_test/255

        # convert class vectors to binary class matrices
        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)


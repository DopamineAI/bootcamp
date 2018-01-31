from tempfile import gettempdir
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adadelta
import tensorflow as tf
from uuid import uuid4
import os
from os.path import join
from mnist import MnistDataFiltered
import numpy as np

__author__ = 'uyerushalmi'


class DigitClassifier(object):
    batch_size = 128
    epochs = 12
    layer_1_size = 32
    layer_2_size = 64
    layer_3_size = 128

    def __init__(self, test_filter):
        self.mnist_data = MnistDataFiltered(0,0,test_filter)
        self.num_classes = self.mnist_data.num_classes
        self.x_test = self.mnist_data.x_test
        self.y_test = self.mnist_data.y_test

    def save(self, path):
        with self.graph.as_default():
            self.model.save(path)

    def load(self, path):
        with self.graph.as_default():
            self.model = load_model(path)

    def build_graph(self):
        self.model = Sequential()

        self.model.add(Conv2D(DigitClassifier.layer_1_size, kernel_size=(3, 3), trainable = True,
                         activation='relu',
                         input_shape=self.mnist_data.input_shape))
        self.model.add(Conv2D(DigitClassifier.layer_2_size, (3, 3), activation='relu', trainable = True))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(DigitClassifier.layer_3_size, activation='relu'))
        self.model.add(Dropout(0.5))

        # self.model.add(Flatten(input_shape=DigitClassifier.mnist_data.input_shape))

        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def freeze_first_layers(self):
        self.model.layers[0].trainable = False
        self.model.layers[1].trainable = True
        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def predict(self, data):
        with self.graph.as_default():
            return self.model.predict(data, verbose=0)

    def predict_classes(self, data):
        return np.argmax(self.predict(data),1)

    def evaluate(self):
        with self.graph.as_default():
            score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score

    def train(self, x_train, y_train, epochs=1):
        with self.graph.as_default():
            self.model.fit(x_train, y_train,
                           batch_size=DigitClassifier.batch_size,
                           epochs=epochs,
                           verbose=0,
                           validation_data=(self.x_test, self.y_test))

    def train_and_return_accuracy_change(self, x_train, y_train):
        with self.graph.as_default():
            score_before = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            score_after = score_before
            tempfile = join(gettempdir(), str(uuid4()))
            self.save(tempfile)
            try:
                while True:
                    self.train(x_train, y_train)
                    last_score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                    if last_score[1] <= score_after[1]:
                        break
                    score_after = last_score
            finally:
                score_after = last_score
                if score_before[1] > score_after[1]:
                    self.load(tempfile)
                    score_after = score_before	
                os.remove(tempfile)
            return score_before, score_after


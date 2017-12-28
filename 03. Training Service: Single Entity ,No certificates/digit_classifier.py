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
from mnist import MnistData


__author__ = 'uyerushalmi'


class DigitClassifier(object):
    batch_size = 128
    epochs = 12
    layer_1_size = 32
    layer_2_size = 64
    layer_3_size = 128
    first_available_train_samples = 1000

    mnist_data = MnistData(0,0)
    num_classes = mnist_data.num_classes
    x_test = mnist_data.x_test
    y_test = mnist_data.y_test

    def save(self, path):
        with self.graph.as_default():
            self.model.save(path)

    def load(self, path):
        with self.graph.as_default():
            self.model = load_model(path)

    def build_graph(self):
        self.model = Sequential()

        self.model.add(Conv2D(DigitClassifier.layer_1_size, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=DigitClassifier.mnist_data.input_shape))
        self.model.add(Conv2D(DigitClassifier.layer_2_size, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(DigitClassifier.layer_3_size, activation='relu'))
        self.model.add(Dropout(0.5))

        # self.model.add(Flatten(input_shape=DigitClassifier.mnist_data.input_shape))

        self.model.add(Dense(DigitClassifier.num_classes, activation='softmax'))

        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def predict(self, data):
        with self.graph.as_default():
            return self.model.predict(data, verbose=0)

    def evaluate(self):
        with self.graph.as_default():
            score = self.model.evaluate(DigitClassifier.x_test, DigitClassifier.y_test, verbose=0)

    def train(self, x_train, y_train):
        with self.graph.as_default():
            self.model.fit(x_train, y_train,
                           batch_size=DigitClassifier.batch_size,
                           epochs=1,
                           verbose=0,
                           validation_data=(DigitClassifier.x_test, DigitClassifier.y_test))

    def train_and_return_accuracy_change(self, x_train, y_train):
        with self.graph.as_default():
            score_before = self.model.evaluate(DigitClassifier.x_test, DigitClassifier.y_test, verbose=0)
            score_after = score_before
            tempfile = join(gettempdir(), str(uuid4()))
            self.save(tempfile)
            try:
                while True:
                    self.train(x_train, y_train)
                    last_score = self.model.evaluate(DigitClassifier.x_test, DigitClassifier.y_test, verbose=0)
                    if last_score[1] > score_after[1]:
                        score_after = last_score
                    else:
                        break
            finally:
                if score_before[1] > score_after[1]:
                    self.load(tempfile)
                os.remove(tempfile)
            return score_before, score_after


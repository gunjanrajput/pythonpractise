import numpy as np 
import mnist #Get data set from 
import matplotlib.pyplot as plt #Graph
import tensorflow as tf
import timeit
from keras.models import Sequential #ANN architecture
from keras.layers import Dense #The layers in the ANN
from keras.utils import to_categorical
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

#### custom AF ####
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from try_5feb import gunjan_activation
get_custom_objects().update({'gunjan_activation': Activation(gunjan_activation)})


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
network = models.Sequential()
network.add(layers.Dense(784, activation='gunjan_activation', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# network = models.Sequential()
# network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))                
# network.compile(optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=50, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
plot(train_images, train_labels)
print('test_acc:', test_acc, 'test_loss', test_loss)
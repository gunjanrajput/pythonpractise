use_tf_keras = True

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


if use_tf_keras:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.utils import to_categorical
else:
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
#train_images = mnist.train_images()
#train_labels = mnist.train_labels()
#test_images = mnist.test_images()
#test_labels = mnist.test_labels()
# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])
print(train_labels)
loss_name = 'categorical_crossentropy'

# Compile the model.
model.compile(
  optimizer='adam',
  loss=loss_name,
  metrics=['accuracy'],
)
# This causes an error after switching to the tf.keras
for layer in model.layers:
    loss = tf.reduce_mean(layer.output, keepdims=True)
    model.metrics.append(loss)
    #model.metrics_tensors.append(loss)  # if tf.keras: AttributeError: 'Sequential' object has no attribute 'metrics_tensors'

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  batch_size=32,
)
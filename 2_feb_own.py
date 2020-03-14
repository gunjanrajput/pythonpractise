import numpy as np 
import mnist #Get data set from 
import matplotlib.pyplot as plt #Graph
from keras.models import Sequential #ANN architecture
from keras.layers import Dense #The layers in the ANN
from keras.utils import to_categorical
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.layers import Dense

#### custom AF ####
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def gunjan_activation(x):
    return (K.sigmoid(x) * 5) - 1

get_custom_objects().update({'gunjan_activation': Activation(gunjan_activation)})
# Usage
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation(gunjan_activation, name='gunjan_activationsssss'))
print(model.summary())

train_images = mnist.train_images() #training data images
train_labels = mnist.train_labels() #training data labels
test_images = mnist.test_images() #training data images
test_labels = mnist.test_labels() #training data labels
#Normalize the images. Normalize the pixel values from [0, 255] to
# [-0.5 , 0.5] to make our network easier to train

############################ OWn tanh function

################################
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5
#Flatten the images. Flatten each 28x28 image into a 28^2 = 784 dimensional vector 
#to pass into the neural network
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))
#Print the shape 
print(train_images.shape)# 60,000 rows and 784 cols
print(test_images.shape) # 10,000 rows and 784 cols
#Build the model
# 3 layers, 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and softmax function
model = Sequential()
#model.add( Dense(64, activation='tanh1', input_dim=784))
model.add( Dense(64, activation='gunjan_activation'))
model.add( Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
#Compile the model
#The loss function measures how well the model did on training , and then tries 
#to improve on it using the optimizer
model.compile(
  optimizer='adam',
    loss = 'categorical_crossentropy', #(classes that are greater than 2) 
    metrics = ['accuracy']
)
#Train the model
model.fit(
  train_images,
    to_categorical(train_labels), # Ex. 2 it expects [0, 0 ,1,0, 0, 0,0,0,0,0]
    epochs = 50, #The number of iterations over the entire dataset to train on
    batch_size=32 #the number of samples per gradient update for training
)
#Evaluate the model
model.evaluate(
  test_images, 
  to_categorical(test_labels)
)
#model.save_weights('model.h5')
#predict on the first 5 test images
predictions = model.predict(test_images[:5])
#print our models prediction 
print(np.argmax(predictions, axis = 1))
print(test_labels[:5])
import matplotlib.pyplot as plt
for i in range(0,5):
  first_image = test_images[i]
  first_image = np.array(first_image, dtype='float')
  pixels = first_image.reshape((28, 28))
  plt.imshow(pixels, cmap='gray')
  plt.show()
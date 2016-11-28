import tensorflow as tf
import pickle
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import math
import random
#from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variables import Variable


# TODO: Implement load the data here.
# TODO: fill this in based on where you saved the training and testing data


batch_size = 128
nb_epoch = 10 

training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# STOP: Do not change the tests below. Your implementation should pass these tests. 
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."

# TODO: Implement data normalization here.

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def normalize_image_data(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.01, 0.99]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    min = np.min(image_data)
    max = np.max(image_data)
    return a + ( ( (image_data - min)*(b - a) )/( max - min ) )


X_train = normalize_image_data(X_train)
X_test  = normalize_image_data(X_test)

y_train = to_categorical(y_train,43)
y_test  = to_categorical(y_test,43)

# STOP: Do not change the tests below. Your implementation should pass these tests. 
assert(round(np.mean(X_train)) == 0), "The mean of the input data is: %f" % np.mean(X_train)
assert(np.min(X_train) == -0.5 and np.max(X_train) == 0.5), "The range of the input data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))


kernel_size = (4,4)
pool_size   = (2,2)
nb_filters  = 32

input_shape = (32,32,3)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(43,name="output"))
model.add(Activation('softmax'))


# STOP: Do not change the tests below. Your implementation should pass these tests.
#assert(model.get_layer(name="hidden1").input_shape == (None, 32*32*3)), "The input shape is: %s" % model.get_layer(name="hidden1").input_shape
#assert(model.get_layer(name="output").output_shape == (None, 43)), "The output shape is: %s" % model.get_layer(name="output").output_shape 


#X_train = X_train.reshape(X_train.shape[0], 32*32*3)
#X_test = X_test.reshape(X_test.shape[0], 32*32*3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


#print (history.history['acc'][0])

# STOP: Do not change the tests below. Your implementation should pass these tests.
#assert(history.history['acc'][0] > 0.5), "The training accuracy was: %.3f" % history.history['acc']

X_train, X_val, y_train, y_val  = train_test_split( X_train, y_train, test_size=0.05, random_state=832289)

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

# STOP: Do not change the tests below. Your implementation should pass these tests.
#assert(round(X_train.shape[0] / float(X_val.shape[0])) == 3), "The training set is %.3f times larger than the validation set." % X_train.shape[0] / float(X_val.shape[0])
#assert(history.history['val_acc'][0] > 0.6), "The validation accuracy is: %.3f" % history.history['val_acc'][0]

model.evaluate(X_test, y_test)

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

def to_categorical(y, classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y:       class vector to be converted into a matrix
        classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    '''
    if not classes:
        classes = np.max(y)+1
    Y = np.zeros((len(y), classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def normalize_image_data(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.01, 0.99]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a   = -0.5
    b   = 0.5
    min = np.min(image_data)
    max = np.max(image_data)
    return a + (( (image_data - min)*(b - a))/( max - min ))




batch_size     = 128
nb_epoch       = 1 
kernel1        = (5,5)
kernel2        = (5,5)
kernel3        = (5,5)
kernel4        = (3,3)
kernel5        = (3,3)
pool_size      = (2,2)

filters1       = 24
filters2       = 36
filters3       = 48
filters4       = 64

layer1_density = 1024
layer2_density = 128
layer3_density = 64
layer4_density = 16

training_file  = "data.p"

#X_train    = normalize_image_data(X_train)
#y_train    = to_categorical      (y_train,nb_classes)

with open(training_file, mode='rb') as f:
    train = pickle.load(f)


X_train, y_train = train['features'], train['labels']
X_train 	 = np.array(X_train)
y_train 	 = np.array(y_train)

input_shape      = np.shape(X_train[0])
nb_classes       = np.shape(np.unique(y_train))[0] 

model = Sequential()
model.add(Convolution2D(filters1, kernel1[0], kernel1[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(filters2, kernel2[0], kernel2[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(filters3, kernel3[0], kernel3[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(filters4, kernel4[0], kernel4[1], border_mode='valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))

#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(layer1_density))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(layer2_density))
prediction = model.add(Dense(1))
#model.add(Activation('softmax'))


model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])



X_train, X_val, y_train, y_val  = train_test_split( X_train, y_train, test_size=0.05, random_state=832289)
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

print (history)

json_filename = "model.json"
md5_filename  = "model.h5"

json_file     = open(json_filename,mode='w')
md5_file      = open(md5_filename, mode='w')

json_string   = model.to_json()

#json_file.write(json_string)
json_file.write(json_string)
model.save_weights(md5_filename)


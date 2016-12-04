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

from collections import defaultdict


import matplotlib.image as mpimg
import csv

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variables import Variable


csv_filename = "driving_log.csv"


csv_file  = open (csv_filename)
reader    = csv.reader(csv_file,delimiter=",")


all_files = list(reader)

data = defaultdict(list)


for i in range(len(all_files)):
    image = (mpimg.imread(all_files[i][0])).astype('uint8')
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lab = all_files[i][3]

    data['features'].append(image)
    data['labels'].append(float(lab))
  

   
filehandler = open("data.p","wb")
pickle.dump (data,filehandler)

#filehandler.close()

file = open("data.p","rb")

train = pickle.load(file)
file.close()

X_train = train['features']
y_train = train['labels']


y_train = np.array(y_train)
X_train = np.array(X_train)

positives = len(np.where(y_train > 0)[0])
negatives = len(np.where(y_train < 0)[0])
#zeros     = np.where(y_train ==0)

#print (positives)
#print (negatives)
print (negatives - positives)

target_X = X_train[np.where(y_train > 0)] 
target_y = y_train[np.where(y_train > 0)]

print (y_train.shape)

for i in range(negatives-positives):
   #add_image[i] = random.choice(train_images)
   index = random.randint(0,len(target_X)-1)
   X_train = np.append(X_train,target_X[index])
   y_train = np.append(y_train,target_y[index])


print (y_train.shape)
#print (float(y_train[0]))

#for row in reader:
#   content = list(row[i] 
#batch_size = 128
#nb_epoch = 10 

#training_file = "train.p"
#testing_file = "test.p"

#with open(training_file, mode='rb') as f:
#    train = pickle.load(f)

#with open(testing_file, mode='rb') as f:
#    test = pickle.load(f)

#X_train, y_train = train['features'], train['labels']
#X_test, y_test = test['features'], test['labels']
#
# STOP: Do not change the tests below. Your implementation should pass these tests. 

# TODO: Implement data normalization here.

#def to_categorical(y, nb_classes=None):
#    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
#    # Arguments
#        y: class vector to be converted into a matrix
#        nb_classes: total number of classes
    # Returns
#        A binary matrix representation of the input.
#    '''
#    if not nb_classes:
#        nb_classes = np.max(y)+1
#    Y = np.zeros((len(y), nb_classes))
#    for i in range(len(y)):
#        Y[i, y[i]] = 1.
#    return Y


#def normalize_image_data(image_data):
#    """
#    Normalize the image data with Min-Max scaling to a range of [0.01, 0.99]
#    :param image_data: The image data to be normalized
#    :return: Normalized image data
#    """
#    a = -0.5
#    b = 0.5
#    min = np.min(image_data)
#    max = np.max(image_data)
#    return a + ( ( (image_data - min)*(b - a) )/( max - min ) )
#

#X_train = normalize_image_data(X_train)
#X_test  = normalize_image_data(X_test)

#y_train = to_categorical(y_train,43)
#y_test  = to_categorical(y_test,43)

#kernel_size = (4,4)
#pool_size   = (2,2)
#nb_filters  = 32

#input_shape = (32,32,3)

#model = Sequential()

#model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(43,name="output"))
#model.add(Activation('softmax'))


#X_train = X_train.reshape(X_train.shape[0], 32*32*3)
#X_test = X_test.reshape(X_test.shape[0], 32*32*3)

#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')


#model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


#X_train, X_val, y_train, y_val  = train_test_split( X_train, y_train, test_size=0.05, random_state=832289)

#history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

#model.evaluate(X_test, y_test)

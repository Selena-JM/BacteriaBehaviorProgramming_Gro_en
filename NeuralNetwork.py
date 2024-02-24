import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import matplotlib as mpl

# Data creation 
X = [[0, 0,	0,	0],
    [1,	0,	0,	0],
    [0,	1,	0,	0],
    [0,	0,	1,	0],
    [0,	0,	0,	1],
    [1,	1,	0,	0],
    [1,	0,	1,	0],
    [1,	0,	0,	1],
    [0,	1,	1,	0],
    [0,	1,	0,	1],
    [0,	0,	1,	1],
    [1,	1,	1,	0],
    [1,	1,	0,	1],
    [1,	0,	1,	1],
    [0,	1,	1,	1],
    [1,	1,	1,	1]]

Y = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]

X_test = [X[i] for i in [0,5,9,15]]

Y_test = [Y[i] for i in [0,5,9,15]]



# Defining the model
#see page 4 of the original paper for the different parameters 
# Here we want each category to be mutually incompatible so we add a layer with one neuron that has softmax activation function and 

model = keras.Sequential([keras.layers.Dense(1, input_shape=[4], activation='softmax')])   #keras.layers.Dense(4, activation='relu', input_shape=[4], kernel_regularizer=regularizers.l2(0.0001)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y)

model.evaluate(X_test,Y_test)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn
import itertools
from tensorflow_addons.optimizers import CyclicalLearningRate
import matplotlib as mpl
mpl.style.use('seaborn')

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


# Defining the model

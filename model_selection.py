import csv
import math

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from random import randint
from config import Config
from collections import Counter
import psutil

########################################################################################################################
# GLOBAL VARIABLES                                                                                                     #
########################################################################################################################

# Amount of data to perform first training
TIME_SERIES_BATCH_SIZE = 1000
# Random seeds
SEED = 100
# Window size of previous data to analyse
WINDOW_SIZE = 100
# Loss threshold to be considered as a change in dataset
CHANGE_THRESHOLD = 0.65
# Maximum number of epochs with low loss until retraining
NUMBER_EPOCHS_TO_RETRAIN = 20

LOG = Config(logger_name='Main').Logger

########################################################################################################################
# LOAD DATASET                                                                                                         #
########################################################################################################################

with open('X_hyperplane.txt') as file:
    reader = csv.reader(file)
    x_data = []
    columns = None
    for row in reader:
        x_data.append(row)
        if columns is None:
            columns = [f'x{i}' for i in range(len(row))]
    data = pd.DataFrame(x_data, dtype=float, columns=columns)

with open('y_hyperplane.txt') as file:
    reader = csv.reader(file)
    y_data = []
    for row in reader:
        y_data += row
    data['y'] = pd.Series(y_data, dtype=int)

########################################################################################################################
# ONLINE LEARNING                                                                                                      #
########################################################################################################################

params = [
    {
        'hidden_layer_sizes': [5, 10, 100, 500],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter': [100, 1000, 2000, 10000],
        'early_stopping': [True],
        'n_iter_no_change': [2, 10, 100],
        'validation_fraction': [0.05, 0.1, 0.2, 0.3],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [10, 100, 200],
        'learning_rate_init': [0.001, 0.01, 0.05],
        'tol': [1e-6, 1e-4, 1e-2]
    },
    {
        'hidden_layer_sizes': [5, 10, 100, 500],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter': [100, 1000, 2000, 10000],
        'early_stopping': [False],
        'n_iter_no_change': [2, 10, 100],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [10, 100, 200],
        'learning_rate_init': [0.001, 0.01, 0.05],
        'tol': [1e-6, 1e-4, 1e-2]
    }
]

params2 = {
        'hidden_layer_sizes': [5, 10, 100, 500],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter': [100, 1000, 2000, 10000],
        'early_stopping': [True],
        'n_iter_no_change': [2, 10, 100],
        'validation_fraction': [0.05, 0.1, 0.2, 0.3],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [10, 100, 200],
        'learning_rate_init': [0.001, 0.01, 0.05],
        'tol': [1e-6, 1e-4, 1e-2]
    }

clf = GridSearchCV(MLPClassifier(), param_grid=params2, verbose=5, n_jobs=6)

batch_size = int(math.ceil(data.shape[0] / TIME_SERIES_BATCH_SIZE))

clf.fit(data.iloc[0:TIME_SERIES_BATCH_SIZE, 0:-1], data.iloc[0:TIME_SERIES_BATCH_SIZE, -1])

print(clf.best_params_)

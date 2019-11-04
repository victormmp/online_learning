import csv
import math

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from online_mlp import OMLP
from random import randint
from config import Config
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

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
CHANGE_THRESHOLD = 0
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

model = OMLP(
    seed=SEED,
    window_size=WINDOW_SIZE,
    threshold=CHANGE_THRESHOLD,
    n_epochs_to_retrain=NUMBER_EPOCHS_TO_RETRAIN
)

batch_size = int(math.ceil(data.shape[0]/TIME_SERIES_BATCH_SIZE))

model.read_batch(batch=data[0:TIME_SERIES_BATCH_SIZE], first_batch=True)
number_of_observations = data.shape[0]

scores = []
retrainings = []
epoch = 0
for i in range(TIME_SERIES_BATCH_SIZE, number_of_observations):
    sample = data.iloc[i]
    y = sample[-1]
    prediction, window_score, change_happened = model.read_sample(sample=sample)
    LOG.info(f'[{i} / {number_of_observations}] Window Score (window size is {WINDOW_SIZE} samples):'
             f' {window_score:0.4f}')
    scores.append([epoch, window_score])
    if change_happened:
        retrainings.append(epoch)
    epoch += 1

LOG.info(f'Number of trainings: {len(model.training_scores)}')
LOG.info(f'Mean test score after trainings: {np.mean(model.training_scores)}')

scores = pd.DataFrame(scores, columns=['Epochs', 'Accuracy'])
plt.figure()
FONT_SIZE = 24
plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
sns.set(style='darkgrid')
for r in retrainings:
    plt.axvline(r, 0, 1, linestyle='--', color='red')
sns.lineplot(data=scores, x='Epochs', y='Accuracy')
LOG.info(f"max score = {scores['Accuracy'].max()} min scores = {scores['Accuracy'].min()}, mean = {scores['Accuracy'].mean()} std = {scores['Accuracy'].std()}")
plt.title('Curva de Loss')
plt.show()


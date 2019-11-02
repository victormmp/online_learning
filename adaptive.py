import csv
import math

import pandas as pd
from sklearn.metrics import average_precision_score
from online_mlp import OMLP
from random import randint
from config import Config
from collections import Counter

TIME_SERIES_BATCH_SIZE = 200
SEED = 100
LOG = Config(logger_name='Main').Logger

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

# Implement a time sampling as an iterator over the
# original data frame.
    
# initial_batch = data[0:TIME_SERIES_BATCH_SIZE]
# data = data[TIME_SERIES_BATCH_SIZE:]
#
# for i, sample in data.iterrows():
#     continue

model = OMLP(seed=SEED)

batch_size = int(math.ceil(data.shape[0]/TIME_SERIES_BATCH_SIZE))

model.read_batch(batch=data[0:TIME_SERIES_BATCH_SIZE], first_batch=True)

for i in range(TIME_SERIES_BATCH_SIZE, data.shape[0], TIME_SERIES_BATCH_SIZE):
    batch = data[i:i+TIME_SERIES_BATCH_SIZE]
    y = batch.iloc[:, -1].values
    prediction = model.read_batch(batch=batch)
    score = average_precision_score(y, prediction)
    LOG.info(f'Classes on real: {Counter(y)} - Classes on predicted: {Counter(prediction)} - Score: {score:0.4f}')

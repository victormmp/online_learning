# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:20:12 2019

@author: victor
"""

from typing import Dict, List

from config import Config

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


LOG = Config(logger_name='OMLP').Logger

class OMLP:
    def __init__(
            self,
            seed: int = None,
            model_parameters: Dict = None,
            threshold: float = 0.6,
            epochs: int = 10000,
            window_size: int = 100,
            n_epochs_to_retrain: int = 20,
            *arg: List,
            **kwarg: Dict):
        """
        Model for Online learning Multilayer Perceptron.
        Args:
            seed: Random seed.
            model_parameters: MLP Parameters, according to sklearn.neural_network.MLPClassifier documentation.
            threshold: Limit value to be considered as a change in the dataset.
            epochs: Number of epochs of the training.
            window_size: Number of consecutive detections with low loss value to be performed until
                            new training is requested.
            n_epochs_to_retrain: Maximum number of epochs with low loss until retraining.
            *arg:
            **kwarg:
        """

        self.seed = seed
        self.n_epochs = epochs
        if model_parameters:
            self.model_parameters = model_parameters
        else:
            self.model_parameters = {
                'hidden_layer_sizes': (100,),
                'activation': 'tanh',
                'max_iter': self.n_epochs,
                'random_state': self.seed,
                'verbose': False,
                'early_stopping': True,
                'n_iter_no_change': 10,
                'validation_fraction': 0.1
            }

        self.model: MLPClassifier = MLPClassifier(**self.model_parameters)
        self.previous_batch = pd.DataFrame([])
        self.threshold = threshold
        self.window_size = window_size
        self.n_epochs_to_retrain = n_epochs_to_retrain
        self.curr_turn = 0
        self.training_scores = []

    def build_mlp(self, **model_parameters):
        self.model_parameters: Dict = model_parameters
        self.model: MLPClassifier = MLPClassifier(**model_parameters)
        return self

    def train_mlp(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
        self.model.fit(x_train, y_train)
        score = self.model.score(x_test, y_test)
        LOG.info(f'Training test score: {score}')
        self.training_scores.append(score)
        return self

    def predict(self, x):
        if isinstance(x, pd.Series):
            return self.model.predict([x])
        elif isinstance(x, pd.DataFrame):
            return self.model.predict(x)
        else:
            LOG.error(f'Not supported type {type(x)}.')

    def read_batch(self, batch: pd.DataFrame, first_batch=False):

        # Update the previous batch considering the configured window size. Keep just the newer ones.
        if batch.shape[0] < self.window_size:
            self.previous_batch.append(batch)
        else:
            self.previous_batch = batch

        # Adjust the previous batch size to be within the window size
        if self.previous_batch.shape[0] > self.window_size:
            self.previous_batch = \
                self.previous_batch.iloc[self.previous_batch.shape[0]-self.window_size: self.previous_batch.shape[0]]

        x = batch.iloc[:, 0:-1]
        y = None
        if 'y' in batch.columns:
            y = batch.iloc[:, -1].values
        if y is None:
            LOG.error(f'Y value not set! The data frame columns are {list(batch.columns)}')
            return None
        if first_batch:
            self.curr_turn = 0
            if y.any():
                self.train_mlp(x, y)
            return None

        prediction = self.predict(x)

        if self.change_detection(prediction, y, self.threshold):
            if y.any():
                self.train_mlp(x, y)
            else:
                LOG.info('No label were informed.')

        return prediction

    def read_sample(self, sample: pd.Series()):
        x = sample[0:-1]
        y = None
        if 'y' in list(sample.keys()):
            y = sample['y']
        if y is None:
            LOG.error(f'Y value not set! The data frame columns are {list(sample.keys())}.')
            return None

        # Get label prediction for the current sample
        prediction_sample = self.predict(x)

        # Update saved dataset window, removing the oldest one and attaching the new sample
        self.previous_batch = self.previous_batch.append(sample)
        if self.previous_batch.shape[0] > self.window_size:
            self.previous_batch = \
                self.previous_batch.iloc[self.previous_batch.shape[0] - self.window_size: self.previous_batch.shape[0]]

        # Get the prediction for the updated window with the current model
        prediction = self.predict(self.previous_batch.iloc[:, 0:-1])

        # Use the window prediction to detect dataset shift
        change_happened, window_score = self.change_detection(prediction, self.previous_batch['y'], self.threshold)
        if change_happened:
            if y.any():
                self.train_mlp(self.previous_batch.iloc[:, 0:-1], self.previous_batch['y'])
            else:
                LOG.info('No label were informed.')

        return prediction_sample, window_score, change_happened

    def change_detection(self, prediction, reference, threshold):
        score = self.get_score(reference, prediction)
        if score <= threshold:
            self.curr_turn += 1
        elif self.curr_turn > 0:
            self.curr_turn -= 1

        if self.curr_turn >= self.n_epochs_to_retrain:
            self.curr_turn = 0
            LOG.warning(f'Detected change in dataset, retraining model. Score = {score}')
            return True, score
        return False, score

    @staticmethod
    def get_score(reference, prediction):
        return accuracy_score(reference, prediction)

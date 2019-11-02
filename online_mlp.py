# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:20:12 2019

@author: victor
"""

from typing import Dict, List

from config import Config

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
import warnings

warnings.filterwarnings("ignore")


LOG = Config(logger_name='OMLP').Logger

class OMLP:
    def __init__(
            self,
            seed: int = None,
            model_parameters: Dict = None,
            threshold: float = 0.6,
            epochs: int = 10000,
            *arg: List,
            **kwarg: Dict):
        """
        Model for Online learning Multilayer Perceptron.
        Args:
            seed: Random seed.
            model_parameters: MLP Parameters, according to sklearn.neural_network.MLPClassifier documentation.
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
                'activation': 'relu',
                'max_iter': self.n_epochs,
                'random_state': self.seed,
                'verbose': False,
            }

        self.model: MLPClassifier = MLPClassifier(**self.model_parameters)
        self.previous_batch = pd.DataFrame([])
        self.threshold = threshold

    def build_mlp(self, **model_parameters):
        self.model_parameters: Dict = model_parameters
        self.model: MLPClassifier = MLPClassifier(**model_parameters)
        return self

    def train_mlp(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def read_batch(self, batch: pd.DataFrame, first_batch=False):
        self.previous_batch = batch
        x = batch.iloc[:, 0:-1]
        y = None
        if 'y' in batch.columns:
            y = batch.iloc[:, -1].values
        if y is None:
            LOG.error(f'Y value not set! The data frame columns are {list(batch.columns)}')
            return None
        if first_batch:
            if y.any():
                self.train_mlp(x, y)
            return None

        prediction = self.predict(x)
        score = self.get_score(y, prediction)

        if self.change_detection(prediction, y, self.threshold):
            LOG.warning(f'Detected change in dataset, retraining model. Score = {score}')
            if y.any():
                self.train_mlp(x, y)
            else:
                LOG.info('No label were informed.')

        return prediction

    def change_detection(self, prediction, reference, threshold):
        score = self.get_score(reference, prediction)
        return True if score <= threshold else False

    @staticmethod
    def get_score(reference, prediction):
        return average_precision_score(reference, prediction)

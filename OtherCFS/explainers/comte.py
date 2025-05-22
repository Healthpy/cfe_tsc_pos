import numpy as np
import torch
import torch.nn.functional as F
from tslearn.neighbors import KNeighborsTimeSeries
from .base import BaseCounterfactual
import time
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import logging
import random
import numbers
import multiprocessing
import matplotlib.pyplot as plt
## License: BSD-3-Clause
# https://github.com/peaclab/CoMTE/blob/6042b267928415bbae84448faf2c8b670434cc15/explainers.py

class CoMTE(BaseCounterfactual):
    def __init__(self, model, data_name=None, window_size=10, stride=1, num_distractors=2):
        super().__init__(model)
        self.data_name = data_name
        self.window_size = window_size
        self.stride = stride
        self.num_distractors = num_distractors

    def _find_distractor(self, x, target_class, X_train, y_train):
        """Find distractor using KNN"""
        knn = KNeighborsTimeSeries(n_neighbors=1, metric="dtw")
        target_instances = X_train[y_train == target_class]
        
        if len(target_instances) == 0:
            return None
            
        knn.fit(target_instances)
        _, indices = knn.kneighbors(x.reshape(1, x.shape[0], x.shape[1]))
        return target_instances[indices[0][0]]

    def _evaluate_segment_swap(self, x_current, distractor, segment, target_class):
        """Evaluate improvement from swapping a segment"""
        x_new = x_current.copy()
        x_new[:, segment[0]:segment[1]] = distractor[:, segment[0]:segment[1]]
        probs = self.model.predict_proba(x_new)
        return probs[0][target_class], x_new

    def _optimize_counterfactual(self, x, distractor, target_class):
        """Greedy optimization approach"""
        segments = [(i, i + self.window_size) 
                   for i in range(0, x.shape[1] - self.window_size + 1, self.stride)]
        
        current_cf = x.copy()
        best_prob = self.model.predict_proba(current_cf)[0][target_class]
        improved = True
        
        while improved:
            improved = False
            best_segment = None
            best_new_cf = None
            
            for segment in segments:
                prob, new_cf = self._evaluate_segment_swap(
                    current_cf, distractor, segment, target_class)
                
                if prob > best_prob:
                    best_prob = prob
                    best_segment = segment
                    best_new_cf = new_cf
                    improved = True
            
            if improved:
                current_cf = best_new_cf
                
                # Early stopping if target class achieved
                if np.argmax(self.model.predict_proba(current_cf)[0]) == target_class:
                    break
        
        return current_cf if improved else None

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual explanation"""
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided")
            
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = get_target_class(self.model, x, X_train, y_train)
            
        # Find distractor instance
        distractor = self._find_distractor(x, target_class, X_train, y_train)
        if distractor is None:
            return None
            
        # Optimize counterfactual
        cf = self._optimize_counterfactual(x, distractor, target_class)
        
        if cf is None:
            return None
            
        return cf.reshape(1, cf.shape[0], cf.shape[1])

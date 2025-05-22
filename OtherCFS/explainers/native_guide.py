import numpy as np
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import time

# Here are the key steps in developing the NG_CF (Native Guide Counterfactual) algorithm as outlined in the document:

# 1. **Retrieve Native Guide**:
#    - Given a query time series \( T_q \), find a counterfactual instance \( T_0^{Native} \) in the case-base.
#    - Typically, this instance is the nearest unlike neighbor (NUN) \( T_0^{NUN} \).

# 2. **Adapt Native Guide to Generate Counterfactual**:
#    - Perturb \( T_0^{Native} \) towards \( T_q \) to generate a more proximate counterfactual \( T_0 \).
#    - Use feature-weight vectors \( \omega \) from techniques like Class Activation Mapping (CAM) to identify and modify contiguous, discriminative subsequences of the time series.

# 3. **Ensure Proximity, Sparsity, Plausibility, and Diversity**:
#    - **Proximity**: Measure closeness using distance metrics like Manhattan distance.
#    - **Sparsity**: Modify only a single discriminative portion of the time series.
#    - **Plausibility**: Ensure the counterfactual is within the data distribution.
#    - **Diversity**: Generate multiple diverse explanations using different in-sample counterfactuals.

# These steps ensure that the generated counterfactual explanations are informative, plausible, and useful for understanding time series classifications.

class NG(BaseCounterfactual):
    """Native Guide (Nearest Unlike Neighbor) counterfactual method."""
    def __init__(self, model, data_name=None, distance='dtw'):
        super().__init__(model)
        self.data_name = data_name
        self.distance = distance
    
    def _find_nun(self, x, target_class, X_train, y_train):
        """Find nearest unlike neighbor."""
        target_indices = np.where(y_train == target_class)[0]
        if not len(target_indices):
            return None
            
        X_target = X_train[target_indices]
        distances = []
        x_flat = x.flatten()
        
        for x_t in X_target:
            x_t_flat = x_t.flatten()
            dist = (dtw.distance(x_flat, x_t_flat) if self.distance == 'dtw' 
                   else euclidean(x_flat, x_t_flat))
            distances.append(dist)
            
        return X_target[np.argmin(distances)]
    
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual using NUN approach."""
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train required")
            
        # Ensure 2D shape
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        # Find nearest unlike neighbor
        nun = self._find_nun(x, target_class, X_train, y_train)
        if nun is None:
            return None
            
        if len(nun.shape) == 3:
            nun = nun.reshape(nun.shape[1], nun.shape[2])
            
        # Generate counterfactual
        cf = x.copy()
        changed = False
        
        for i in range(x.shape[1]):
            temp_cf = cf.copy()
            temp_cf[:, i] = nun[:, i]
            
            if self._is_valid_cf(temp_cf, x, target_class):
                cf[:, i] = nun[:, i]
                changed = True
                
        return cf.reshape(1, cf.shape[0], cf.shape[1]) if changed else None

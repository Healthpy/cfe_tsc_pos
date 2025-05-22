import numpy as np
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
from utils.neighbors import KNeighborsTimeSeries
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import os
import time

class AB_CF(BaseCounterfactual):
    """Attention-based counterfactual method."""
    def __init__(self, model, data_name=None, n_segments=10, window_size_ratio=0.1):
        super().__init__(model, data_name)
        self.n_segments = n_segments
        self.window_size_ratio = window_size_ratio

    def sliding_window_3d(self, data, window_size, stride):
        """Extract subsequences using sliding window."""
        num_features, num_timesteps = data.shape
        num_subsequences = ((num_timesteps - window_size) // stride) + 1
        subsequences = np.zeros((num_subsequences, num_features, window_size))
        
        for j in range(num_subsequences):
            start = j * stride
            end = start + window_size
            subsequences[j] = data[:, start:end]
            
        return subsequences
    
    def entropy(self, predict_proba):
        """Compute prediction entropy using scipy.stats."""
        return entropy(predict_proba) if np.any(predict_proba) else 0.0
    
    def native_guide_retrieval(self, query, target_label, X_train, y_train):
        """Find nearest neighbor of target class."""
        try:
            # Convert target_label to integer if it's a numpy array
            if isinstance(target_label, np.ndarray):
                target_label = target_label.item()
            target_label = int(target_label)
            
            # Ensure query has correct shape
            if len(query.shape) == 2:
                query = query.reshape(1, query.shape[0], query.shape[1])
                
            # Convert y_train to 1D array if needed
            if len(y_train.shape) > 1:
                y_train = np.argmax(y_train, axis=1)
                
            # Get target class samples
            target_indices = np.where(y_train == target_label)[0]
            if len(target_indices) == 0:
                print(f"No samples found for target class {target_label}")
                return None
                
            # Prepare target samples for KNN
            target_samples = X_train[target_indices]
            if len(target_samples.shape) != 3:
                target_samples = target_samples.reshape(len(target_samples), 1, -1)
                
            # Initialize and fit KNN
            knn = KNeighborsTimeSeries(n_neighbors=1, metric='dtw')
            knn.fit(target_samples)
            
            # Find nearest neighbor
            _, ind = knn.kneighbors(query)
            nearest_idx = int(target_indices[int(ind[0][0])])
            
            return nearest_idx
            
        except Exception as e:
            print(f"Error in native_guide_retrieval:")
            print(f"Query shape: {query.shape}")
            print(f"Target label: {target_label} (type: {type(target_label)})")
            print(f"y_train shape: {y_train.shape}")
            print(f"Target indices: {target_indices if 'target_indices' in locals() else 'Not created'}")
            raise e

    def _is_valid_cf(self, cf, x, target_class):
        """Check if counterfactual is valid.
        A valid counterfactual should:
        1. Be classified as target class
        2. Be different from original instance
        """
        # Ensure cf is properly shaped for prediction
        if len(cf.shape) == 2:
            cf = cf.reshape(1, cf.shape[0], cf.shape[1])
            
        # Get prediction for counterfactual
        cf_pred = np.argmax(self.predict_proba(cf))
        
        # Check if prediction matches target class and cf is different from x
        return (cf_pred == target_class) and not np.array_equal(cf, x)

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate a counterfactual for the instance x."""
        try:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train must be provided for AB_CF")
                
            # Ensure x is 2D (features, time_steps)
            if len(x.shape) == 3:
                x = x.reshape(x.shape[1], x.shape[2])
                
            # Convert target_class to integer if it's a numpy array
            if isinstance(target_class, np.ndarray):
                target_class = target_class.item()
            elif target_class is None:
                target_class = get_target_class(self.model, x, y_train)
                
            # Calculate window size and stride
            window_size = max(1, int(x.shape[1] * self.window_size_ratio))
            stride = window_size
            
            # Extract subsequences
            subsequences = self.sliding_window_3d(x, window_size, stride)
            
            # Get total time series length from training data
            ts_length = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
            
            # Pad subsequences
            padded_subsequences = np.pad(
                subsequences,
                ((0, 0), (0, 0), (0, ts_length - subsequences.shape[2])),
                mode='constant'
            )
            
            # Calculate entropy for each subsequence
            predict_proba = self.predict_proba(padded_subsequences)
            entropies = np.array([self.entropy(p) for p in predict_proba])
            
            # Sort subsequences by entropy
            indices = np.argsort(entropies)[:self.n_segments]
            
            # Find nearest neighbor
            nun_idx = self.native_guide_retrieval(x, target_class, X_train, y_train)
            if nun_idx is None:
                return None
                
            # Get the nearest unlike neighbor
            nun = X_train[nun_idx]
            if len(nun.shape) == 3:
                nun = nun.reshape(nun.shape[1], nun.shape[2])
                
            # Create counterfactual
            cf = x.copy()
            
            # Try replacing segments
            for idx in indices:
                start = int(idx * stride)
                end = int(start + window_size)
                
                # Ensure indices are within bounds
                if end > cf.shape[1]:
                    end = cf.shape[1]
                
                # Replace segment
                cf[:, start:end] = nun[:, start:end]
                
                # Check if valid
                if self._is_valid_cf(cf, x, target_class):
                    cf_pred = np.argmax(self.predict_proba(cf.reshape(1, cf.shape[0], cf.shape[1])))
                    return cf.reshape(1, cf.shape[0], cf.shape[1]), cf_pred
                    
            return None
            
        except Exception as e:
            raise e

    def explain(self, x, target_class=None):
        """Generate counterfactual explanation for input x.
        Args:
            x: Input time series (shape: [1, n_features, seq_length])
            target_class: Optional target class (if None, second most probable class is used)
        Returns:
            cf: Counterfactual example
            cf_label: Predicted class for counterfactual
        """
        # Load training data from class attributes
        X_train = getattr(self, 'X_train', None)
        y_train = getattr(self, 'y_train', None)
        
        # Generate counterfactual
        result = self.generate(x, target_class, X_train, y_train)
        
        if result is not None:
            cf, cf_label = result
            return cf, cf_label
        return None, None

    def predict(self, x):
        """Get class predictions for x."""
        return np.argmax(self.predict_proba(x), axis=1)
    
    def predict_proba(self, x):
        """Get probability predictions for x."""
        if isinstance(x, np.ndarray):
            if len(x.shape) == 2:
                x = x.reshape(1, x.shape[0], x.shape[1])
        return self.model.predict_proba(x)

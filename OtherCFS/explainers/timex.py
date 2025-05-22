import numpy as np
import torch
import torch.nn.functional as F
from tslearn.barycenters import dtw_barycenter_averaging
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import time

class TimeX(BaseCounterfactual):
    """TimeX: Enhanced with barycenter-based counterfactual generation."""
    
    def __init__(self, model, data_name=None, k_neighbors=5, percentile=90,
                 segment_ratio=0.01, max_iter=100, learning_rate=0.01,
                 dba_max_iter=10, lambda_dba=1.0, noise_std=0.5, 
                 max_segments=5):  # Add max_segments parameter
        super().__init__(model)
        self.data_name = data_name
        self.k_neighbors = k_neighbors
        self.percentile = percentile
        self.segment_ratio = segment_ratio
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.dba_max_iter = dba_max_iter  # Number of DBA iterations
        self.lambda_dba = lambda_dba  # Weight for DBA loss
        self.noise_std = noise_std  # Standard deviation for noise injection
        self.max_segments = max_segments  # Maximum number of segments to modify

    def _compute_saliency(self, x):
        """Compute gradient-based saliency map."""
        x_tensor = torch.tensor(x, requires_grad=True).float()
        output = self.model(x_tensor)
        pred_class = output.argmax().item()
        
        # Get gradients for predicted class
        loss = output[0, pred_class]
        gradients = torch.autograd.grad(loss, x_tensor)[0]
        # print(gradients.shape)
        
        return np.abs(gradients.detach().numpy())
    
    # def _dtw_distance(self, x, y):
    #     """Compute DTW distance between two sequences."""
    #     n, m = len(x), len(y)
    #     dtw_matrix = np.full((n + 1, m + 1), np.inf)
    #     dtw_matrix[0, 0] = 0
        
    #     for i in range(1, n + 1):
    #         for j in range(1, m + 1):
    #             cost = np.linalg.norm(x[i-1] - y[j-1])
    #             dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
    #                                         dtw_matrix[i, j-1],      # deletion
    #                                         dtw_matrix[i-1, j-1])    # match
    #     return dtw_matrix[n, m]

    # def _compute_dba_centroid(self, X):
    #     """Compute Dynamic Barycenter Averaging (DBA) centroid using DTW distance."""
    #     centroid = np.mean(X, axis=0)  # Initial centroid
        
    #     for _ in range(3):  # Number of DBA iterations
    #         # Compute DTW distances instead of Euclidean
    #         distances = np.array([self._dtw_distance(x, centroid) for x in X])
    #         weights = 1 / (distances + 1e-6)  # Avoid division by zero
    #         centroid = np.average(X, weights=weights, axis=0)    
    #     return centroid

    def _compute_loss(self, x_current, x_original, target_class, target_centroid):
        """Compute combined loss with enhanced barycenter guidance."""
        x_tensor = torch.tensor(x_current, requires_grad=True).float()
        output = self.model(x_tensor)
        
        # Compute target class probability
        target_probs = F.softmax(output, dim=1)
        target = torch.tensor([target_class], dtype=torch.long)
        
        # Prediction loss: maximize target class probability
        pred_loss = -torch.log(target_probs[0, target_class] + 1e-6)
        
        # Distance losses
        l1_loss = torch.norm(x_tensor - torch.tensor(x_original).float(), p=1)
        dba_loss = self.lambda_dba * torch.norm(x_tensor - torch.tensor(target_centroid).float(), p=2)
        
        # Weighted combination of losses
        total_loss = pred_loss + l1_loss + dba_loss
        
        return total_loss, x_tensor

    def _find_important_segments(self, saliency, seq_length, segment_size, n_segments):
        """Find multiple important segments based on saliency and return with importance scores."""
        saliency_sum = np.array([np.sum(saliency[:, i:i+segment_size]) 
                                for i in range(seq_length - segment_size + 1)])
        
        # Get top n_segments non-overlapping segments with their importance scores
        segments = []
        scores = []
        temp_saliency = saliency_sum.copy()
        
        for _ in range(n_segments):
            if len(temp_saliency) == 0 or np.all(temp_saliency == -np.inf):
                break
            start = np.argmax(temp_saliency)
            score = temp_saliency[start]
            if score == -np.inf:
                break
                
            segments.append(start)
            scores.append(score)
            
            # Zero out overlapping regions
            start_idx = max(0, start - segment_size)
            end_idx = min(len(temp_saliency), start + segment_size)
            temp_saliency[start_idx:end_idx] = -np.inf
        
        # Sort segments by importance scores
        sorted_idx = np.argsort(scores)[::-1]  # descending order
        sorted_segments = [segments[i] for i in sorted_idx]
        sorted_scores = [scores[i] for i in sorted_idx]
        
        return sorted_segments, sorted_scores

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided for TimeX")
            
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            # Get all possible classes except current prediction
            current_pred = self.model(torch.tensor(x, dtype=torch.float32)).argmax().item()
            possible_classes = np.unique(y_train)
            other_classes = possible_classes[possible_classes != current_pred]
            
            # Select target class with highest probability from other classes
            probs = self.model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            other_probs = probs[0][other_classes]
            target_class = other_classes[np.argmax(other_probs)]
            
        # Get target class examples and compute centroid
        target_examples = X_train[y_train == target_class]
        target_centroid = dtw_barycenter_averaging(target_examples, max_iter=self.dba_max_iter)
        
        # Compute saliency map
        saliency = self._compute_saliency(x)
        
        # Find most influential points
        # threshold = np.percentile(saliency, self.percentile)
        # important_points = np.where(saliency > threshold)
        
        # Get target class examples
        target_examples = X_train[y_train == target_class]
        
        # Compute target class centroid using DBA
        # target_centroid = self._compute_dba_centroid(target_examples)
        target_centroid = dtw_barycenter_averaging(target_examples, max_iter=self.dba_max_iter)
        
        # Initialize perturbation segment
        seq_length = x.shape[1]
        segment_size = max(int(seq_length * self.segment_ratio), 1)
        # print(f"Segment size: {segment_size}")
        
        # # Find most influential contiguous segment
        # saliency_sum = np.array([np.sum(saliency[:, i:i+segment_size]) 
        #                         for i in range(seq_length - segment_size + 1)])
        # segment_start = np.argmax(saliency_sum)
        
        # Optimization loop
        cf = x.copy()
        optimizer = torch.optim.Adam([torch.tensor(cf, requires_grad=True)], 
                                   lr=self.learning_rate)
        
        # Initialize with one segment
        n_active_segments = 1
        seq_length = x.shape[1]
        segment_size = max(int(seq_length * self.segment_ratio), 1)
        
        # Optimization loop
        cf = x.copy()
        best_loss = float('inf')
        steps_without_improvement = 0
        
        for _ in range(self.max_iter):
            # Update segment starts based on current number of active segments
            segment_starts, importance_scores = self._find_important_segments(
                saliency, seq_length, segment_size, n_active_segments)
            
            loss, x_tensor = self._compute_loss(cf, x, target_class, target_centroid)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cf_update = x_tensor.detach().numpy()
            
            # Update all active segments with importance-weighted noise
            for start, importance in zip(segment_starts, importance_scores):
                segment_shape = cf[:, start:start+segment_size].shape
                # Scale noise by importance
                scaled_noise = self.noise_std * (importance / max(importance_scores))
                noise = np.random.normal(0, scaled_noise, size=segment_shape)
                cf[:, start:start+segment_size] = \
                    cf_update[:, start:start+segment_size] + noise
            
            # Check if valid
            if self._is_valid_cf(cf, x, target_class):
                return cf.reshape(1, cf.shape[0], cf.shape[1])
            
            # Adjust strategy if no improvement
            if loss < best_loss:
                best_loss = loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # Every 100 steps, adjust strategy
            if _ % 100 == 0:
                # if steps_without_improvement > 50:
                    # Increase number of segments if not improving
                n_active_segments = min(n_active_segments + 1, self.max_segments)
                steps_without_improvement = 0
            # segment_size = min(segment_size + 1, seq_length)
                self.noise_std *= 0.9
        
        return None

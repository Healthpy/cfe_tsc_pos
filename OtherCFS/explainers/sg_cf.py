import numpy as np
import torch
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import time

class SG_CF(BaseCounterfactual):
    """Shapelet-Guided Counterfactual (SG-CF)."""
    
    def __init__(self, model, data_name=None, min_length=3, max_length=None, n_shapelets=5):
        super().__init__(model)
        self.data_name = data_name
        self.min_length = min_length
        self.max_length = max_length
        self.n_shapelets = n_shapelets
        
    def _extract_shapelets(self, x):
        """Extract discriminative shapelets."""
        n_features, n_timesteps = x.shape
        max_length = self.max_length or n_timesteps // 2
        shapelets = []
        
        # Calculate importance using gradients
        x_tensor = torch.tensor(x, requires_grad=True).float().unsqueeze(0)
        output = self.model(x_tensor)
        pred_class = output.argmax().item()
        grads = torch.autograd.grad(output[0, pred_class], x_tensor)[0]
        importance_map = np.abs(grads.detach().numpy()[0])
        
        # Extract shapelets of different lengths
        for length in range(self.min_length, max_length + 1):
            for start in range(n_timesteps - length + 1):
                shapelet = x[:, start:start+length]
                importance = np.mean(importance_map[:, start:start+length])
                shapelets.append({
                    'shapelet': shapelet,
                    'start': start,
                    'length': length,
                    'importance': importance
                })
                
        # Sort by importance and return top shapelets
        shapelets.sort(key=lambda x: x['importance'], reverse=True)
        return shapelets[:self.n_shapelets]
        
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual using shapelet modifications."""
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = self._get_target_class(x)
            
        # Extract important shapelets
        shapelets = self._extract_shapelets(x)
        cf = x.copy()
        
        # Try different modifications
        modifications = [0.8, 0.9, 1.1, 1.2]
        
        for shapelet_info in shapelets:
            start = shapelet_info['start']
            length = shapelet_info['length']
            orig_shapelet = shapelet_info['shapelet']
            
            for mod in modifications:
                temp_cf = cf.copy()
                temp_cf[:, start:start+length] = orig_shapelet * mod
                
                if self._is_valid_cf(temp_cf, x, target_class):
                    return temp_cf.reshape(1, temp_cf.shape[0], temp_cf.shape[1])
                    
        return None
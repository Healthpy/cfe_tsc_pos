import numpy as np
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class

class DiscoX(BaseCounterfactual):
    """DiscoX: Discord-based counterfactual explanation."""
    
    def __init__(self, model, data_name=None, window_size=10, modification_factor=1.1):
        super().__init__(model)
        self.data_name = data_name
        self.window_size = window_size
        self.modification_factor = modification_factor
        
    def _find_discord(self, x):
        """Find the most discordant subsequence using matrix profile."""
        n_features, n_timesteps = x.shape
        profile = np.zeros(n_timesteps - self.window_size + 1)
        
        for i in range(len(profile)):
            current = x[:, i:i+self.window_size]
            # Calculate distance to all other windows
            distances = []
            for j in range(len(profile)):
                if abs(i-j) >= self.window_size:  # Non-overlapping windows
                    compare = x[:, j:j+self.window_size]
                    dist = np.sqrt(np.sum((current - compare) ** 2))
                    distances.append(dist)
            profile[i] = np.min(distances) if distances else np.inf
            
        return np.argmax(profile)
        
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = self._get_target_class(x)
            
        cf = x.copy()
        discord_idx = self._find_discord(x)
        
        # Try different modifications until valid counterfactual is found
        modifications = [self.modification_factor, 1/self.modification_factor, 
                       -self.modification_factor, -1/self.modification_factor]
        
        for mod in modifications:
            temp_cf = cf.copy()
            temp_cf[:, discord_idx:discord_idx+self.window_size] *= mod
            
            if self._is_valid_cf(temp_cf, x, target_class):
                return temp_cf.reshape(1, temp_cf.shape[0], temp_cf.shape[1])
                
        return None
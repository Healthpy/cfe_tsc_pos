import numpy as np
import torch
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
from utils.timegan_modules import Time_GAN_module, TimeGAN
from utils.timegan_utils import random_generator, extract_time
import time
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform 
from sklearn.ensemble import RandomForestClassifier

class Time_CF(BaseCounterfactual):
    """Time-CF: TimeGAN and shapelet-based counterfactual explanations."""
    
    def __init__(self, model, data_name=None, hidden_dim=64, n_layers=2, 
                 n_epochs=100, batch_size=32, n_shapelets=30):
        super().__init__(model)
        self.data_name = data_name
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_shapelets = n_shapelets
        self.shapelet_transform = RandomShapeletTransform(
            n_shapelet_samples=100, 
            max_shapelets=self.n_shapelets,
            random_state=42
        )
        
        # TimeGAN components
        self.generator = None
        self.discriminator = None
        self.embedder = None
        self.recovery = None
        self.supervisor = None
        self.parameters = None
        
    def _initialize_timegan(self, data):
        """Initialize and train TimeGAN networks."""
        self.parameters = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.n_layers,
            "iterations": 1000,
            "batch_size": self.batch_size,
            "module": "gru",
            "epoch": self.n_epochs
        }
        
        # Train TimeGAN
        self.generator, self.embedder, self.supervisor, \
        self.recovery, self.discriminator, _ = TimeGAN(data, self.parameters)
            
    def _generate_synthetic_data(self, X_train, y_train, target_class):
        """Generate synthetic samples using TimeGAN."""
        # Filter data for target class
        target_indices = np.where(y_train != target_class)[0]
        X_target = X_train[target_indices]
        
        # Generate synthetic samples using TimeGAN
        with torch.no_grad():
            time_info, max_seq_len = extract_time([X_target])
            z = random_generator(self.batch_size, X_target.shape[1], time_info, max_seq_len)
            z_tensor = torch.tensor(z).float()
            
            e_hat, _ = self.generator(z_tensor)
            h_hat, _ = self.supervisor(e_hat)
            synthetic_samples, _ = self.recovery(h_hat)
            
        return synthetic_samples.numpy()

    def _extract_replace_shapelets(self, x, synthetic_samples, shapelets, target_class): 
        """Extract and replace shapelets to generate counterfactuals."""
        counterfactuals = []
        
        for shapelet_info in shapelets:
            start_pos, length = shapelet_info[2], shapelet_info[1]
            
            # Extract synthetic shapelets
            synthetic_segments = []
            for synthetic in synthetic_samples:
                if start_pos + length <= synthetic.shape[1]:
                    segment = synthetic[:, start_pos:start_pos + length]
                    synthetic_segments.append(segment)
            
            # Try replacing segments
            for segment in synthetic_segments:
                cf = x.copy()
                cf[:, start_pos:start_pos + length] = segment
                
                if self._is_valid_cf(cf, x, target_class):
                    counterfactuals.append(cf)
                    
        return counterfactuals if counterfactuals else None

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual using TimeGAN and shapelet-based approach."""
        if X_train is not None and self.generator is None:
            self._initialize_timegan(X_train)
            # Fit shapelet transform
            self.shapelet_transform.fit(X_train, y_train)
            
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = get_target_class(self.model, x, None)
        
        # Generate synthetic samples
        synthetic_samples = self._generate_synthetic_data(X_train, target_class)
        
        # Extract shapelets and generate counterfactuals
        shapelets = self.shapelet_transform.shapelets
        counterfactuals = self._extract_replace_shapelets(x, synthetic_samples, shapelets)
        
        if counterfactuals is None:
            return None
            
        # Return the counterfactual that's most similar to original
        distances = [np.linalg.norm(cf - x) for cf in counterfactuals]
        best_cf = counterfactuals[np.argmin(distances)]
        
        return best_cf.reshape(1, best_cf.shape[0], best_cf.shape[1])

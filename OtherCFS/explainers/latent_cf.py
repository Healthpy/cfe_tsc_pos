import numpy as np
import torch
import torch.nn as nn
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)

class LatentCF(BaseCounterfactual):
    """LatentCF++: Latent space optimization."""
    
    def __init__(self, model, data_name=None, latent_dim=8, lr=0.01, max_iter=100):
        super().__init__(model)
        self.data_name = data_name
        self.latent_dim = latent_dim
        self.lr = lr
        self.max_iter = max_iter
        self.autoencoder = None
        
    def _train_autoencoder(self, X_train):
        """Train autoencoder on training data."""
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train).float()
            
        input_dim = X_train.shape[-1] * X_train.shape[-2]
        X_flat = X_train.reshape(-1, input_dim)
        
        self.autoencoder = AutoEncoder(input_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        
        for _ in range(100):  # Training epochs
            optimizer.zero_grad()
            recon = self.autoencoder(X_flat)
            loss = nn.MSELoss()(recon, X_flat)
            loss.backward()
            optimizer.step()
            
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        if X_train is not None and self.autoencoder is None:
            self._train_autoencoder(X_train)
            
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = self._get_target_class(x)
            
        # Convert to tensor and get initial latent representation
        x_flat = torch.tensor(x.flatten()).float()
        z = self.autoencoder.encode(x_flat)
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=self.lr)
        
        # Optimize in latent space
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            cf_flat = self.autoencoder.decode(z)
            cf = cf_flat.reshape(x.shape)
            
            # Compute loss
            pred = self.model(cf.unsqueeze(0))
            target_prob = pred[0, target_class]
            latent_dist = torch.norm(z - self.autoencoder.encode(x_flat))
            loss = -target_prob + 0.1 * latent_dist
            
            loss.backward()
            optimizer.step()
            
            # Check if valid
            if self._is_valid_cf(cf.detach().numpy(), x, target_class):
                return cf.detach().numpy().reshape(1, cf.shape[0], cf.shape[1])
                
        return None

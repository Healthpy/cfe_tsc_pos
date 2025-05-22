import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.neighbors import NearestNeighbors
# from sklearn.ensemble import IsolationForest
from .base import BaseCounterfactual
from utils.utils import get_target_class
import time

def AutoCELS(model, data_name=None, **kwargs):
    """Automatic CELS/M-CELS selector based on input data dimensionality.
    
    This wrapper automatically selects between CELS (univariate) and M-CELS 
    (multivariate) based on the input data shape.
    
    Args:
        model: The trained model to explain
        data_name: Optional name of the dataset
        **kwargs: Additional arguments passed to both CELS and M-CELS
        
    Example:
        explainer = AutoCELS(model)
        cf = explainer.generate(x, target_class, X_train, y_train)
    """
    base_explainer = U_CELS(model, data_name, **kwargs)
    multivariate_explainer = M_CELS(model, data_name, **kwargs)
    
    class CELS(BaseCounterfactual):
        def __init__(self):
            super().__init__(model, data_name)
            
        def generate(self, x, target_class=None, X_train=None, y_train=None):
            is_multivariate = (len(x.shape) == 3 and x.shape[1] > 1) or (len(x.shape) == 2 and x.shape[0] > 1)
            return multivariate_explainer.generate(x, target_class, X_train, y_train) if is_multivariate \
                   else base_explainer.generate(x, target_class, X_train, y_train)
    
    return CELS()

class U_CELS(BaseCounterfactual):
    """Counterfactual Explainer via Learned Saliency Map (CELS)."""
    def __init__(self, model, data_name=None, learning_rate=0.01, max_iter=100,
                 lambda_valid=1.0, lambda_budget=0.1, lambda_tv=0.1, 
                 tv_beta=2, enable_lr_decay=True, lr_decay=0.9991):
        super().__init__(model)
        self.data_name = data_name
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_valid = lambda_valid
        self.lambda_budget = lambda_budget
        self.lambda_tv = lambda_tv
        self.tv_beta = tv_beta
        self.enable_lr_decay = enable_lr_decay
        self.lr_decay = lr_decay
        self.softmax = nn.Softmax(dim=-1)

    def _tv_norm(self, mask, beta=1):
        """Compute TV norm for regularization."""
        diffs = torch.abs(mask[..., 1:] - mask[..., :-1])
        return torch.mean(torch.pow(diffs, beta))

    def _find_counterfactual_target(self, x):
        """Find the second most likely class as counterfactual target."""
        output = self.softmax(self.model(x.reshape(1, 1, -1).float()))
        return torch.argsort(output, descending=True)[0, 1].item()

    def _learn_saliency(self, x, target_class, nun):
        """Learn saliency map using optimized loss function with early stopping."""
        x_tensor = torch.tensor(x, requires_grad=True).float()
        nun_tensor = torch.tensor(nun, requires_grad=False).float()
        
        # Initialize mask
        mask_init = np.random.uniform(size=[1, x.shape[-1]], low=0, high=1)
        mask = Variable(torch.from_numpy(mask_init), requires_grad=True)
        
        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)
        if self.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.lr_decay)
            
        best_loss = float('inf')
        counter = 0
        max_no_improve = 30
        imp_threshold = 0.001
        
        for i in range(self.max_iter):
            cf_tensor = x_tensor * (1 - mask) + nun_tensor * mask
            output = self.softmax(self.model(cf_tensor.reshape(1, 1, -1).float()))
            
            # Compute losses
            valid_loss = 1 - output[0, target_class]
            budget_loss = torch.mean(torch.abs(mask))
            tv_loss = self._tv_norm(mask, self.tv_beta)
            
            total_loss = (self.lambda_valid * valid_loss + 
                         self.lambda_budget * budget_loss +
                         self.lambda_tv * tv_loss)
            
            if best_loss - total_loss < imp_threshold:
                counter += 1
            else:
                counter = 0
                best_loss = total_loss
                
            if counter >= max_no_improve:
                break
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if self.enable_lr_decay:
                scheduler.step()
                
            mask.data.clamp_(0, 1)
            
        return torch.sigmoid(mask).detach().numpy()

    def _find_nearest_unlike_neighbor(self, x, target_class, X_train, y_train):
        """Find the nearest training instance of the target class."""
        target_instances = X_train[y_train == target_class]
        nbrs = NearestNeighbors(n_neighbors=1).fit(target_instances.reshape(len(target_instances), -1))
        _, indices = nbrs.kneighbors(x.reshape(1, -1))
        return target_instances[indices[0][0]]

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual for univariate time series."""
        # Validate input dimensions
        if (len(x.shape) == 3 and x.shape[1] > 1) or (len(x.shape) == 2 and x.shape[0] > 1):
            raise ValueError("CELS only supports univariate time series. Use M-CELS for multivariate data.")
        
        # Reshape if needed
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
        
        # Find nearest unlike neighbor
        nun = self._find_nearest_unlike_neighbor(x, target_class, X_train, y_train)
        
        # Get optimized saliency
        saliency = self._learn_saliency(x, target_class, nun)
        
        # Apply threshold for sparse perturbations
        threshold = 0.5
        mask = np.where(saliency > threshold, 1, 0)
        
        # Generate counterfactual
        cf = x * (1 - mask) + nun * mask
        
        return cf.reshape(1, cf.shape[0], cf.shape[1])

class M_CELS(BaseCounterfactual):
    """Multivariate Counterfactual Explainer via Learned Saliency Map (M-CELS)."""
    def __init__(self, model, data_name=None, learning_rate=0.01, max_iter=100,
                 lambda_valid=1.0, lambda_sparsity=0.1, lambda_smoothness=0.1, 
                 enable_lr_decay=True, lr_decay=0.9991):
        super().__init__(model)
        self.data_name = data_name
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_valid = lambda_valid
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smoothness = lambda_smoothness
        self.enable_lr_decay = enable_lr_decay
        self.lr_decay = lr_decay
        self.softmax = nn.Softmax(dim=-1)
        
    def compute_saliency(self, x, target_class):
        """Compute initial saliency based on perturbation sensitivity"""
        x_tensor = torch.tensor(x, requires_grad=True).float()
        output = self.softmax(self.model(x_tensor))
        target_prob = output[0, target_class]
        
        grads = torch.autograd.grad(target_prob, x_tensor)[0]
        return torch.abs(grads).detach().numpy()
        
    def compute_loss(self, output, mask, target_class, D, T):
        """Compute total loss with all components"""
        # Validity loss
        valid_loss = self.lambda_valid * (1 - output[0, target_class])
        
        # Sparsity loss across dimensions and time steps
        sparsity_loss = self.lambda_sparsity * torch.mean(torch.abs(mask))
        
        # Smoothness loss for contiguous perturbations
        temp_diff = mask[..., 1:, :] - mask[..., :-1, :]
        dim_diff = mask[..., :, 1:] - mask[..., :, :-1]
        smoothness_loss = self.lambda_smoothness * (
            torch.mean(torch.square(temp_diff)) + 
            torch.mean(torch.square(dim_diff))
        )
        
        return valid_loss + sparsity_loss + smoothness_loss
        
    def _find_nearest_unlike_neighbor(self, x, target_class, X_train, y_train):
        """Find nearest unlike neighbor ensuring interpretability"""
        target_idx = y_train == target_class
        target_samples = X_train[target_idx]
        
        # Reshape for proper distance computation
        x_flat = x.reshape(1, -1)
        samples_flat = target_samples.reshape(len(target_samples), -1)
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(samples_flat)
        _, indices = nbrs.kneighbors(x_flat)
        
        return target_samples[indices[0][0]]
        
    def validate_counterfactual(self, cf, X_train, threshold=-0.5):
        """Validate if counterfactual is within distribution"""
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(random_state=42).fit(
            X_train.reshape(len(X_train), -1))
        score = iso_forest.score_samples(cf.reshape(1, -1))[0]
        return score > threshold
        
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual explanation for multivariate time series."""
        # Validate input dimensions
        if (len(x.shape) == 2 and x.shape[0] == 1) or (len(x.shape) == 3 and x.shape[1] == 1):
            raise ValueError("M-CELS is designed for multivariate time series. Use CELS for univariate data.")
            
        # Ensure proper 3D shape (samples, variables, timesteps)
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
            
        # Initialize saliency map from perturbation sensitivity
        saliency_init = self.compute_saliency(x, target_class)
        mask = Variable(torch.from_numpy(saliency_init), requires_grad=True)
        
        # Find nearest unlike neighbor
        nun = self._find_nearest_unlike_neighbor(x, target_class, X_train, y_train)
        
        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)
        if self.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.lr_decay)
            
        x_tensor = torch.tensor(x, requires_grad=True).float()
        nun_tensor = torch.tensor(nun, requires_grad=False).float()
        
        best_cf = None
        best_loss = float('inf')
        
        for i in range(self.max_iter):
            # Generate counterfactual
            cf_tensor = x_tensor * (1 - mask) + nun_tensor * mask
            output = self.softmax(self.model(cf_tensor))
            
            # Compute loss
            loss = self.compute_loss(output, mask, target_class, 
                                   x.shape[1], x.shape[2])
            
            if loss < best_loss:
                best_loss = loss
                best_cf = cf_tensor.detach().numpy()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.enable_lr_decay:
                scheduler.step()
                
            # Clamp mask values
            mask.data.clamp_(0, 1)
            
            # Check validity
            if output.argmax().item() == target_class:
                cf = cf_tensor.detach().numpy()
                if self.validate_counterfactual(cf, X_train):
                    return cf
                    
        return best_cf if best_cf is not None else None
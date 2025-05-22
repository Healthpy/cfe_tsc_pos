import numpy as np
import torch
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import time

class Glacier(BaseCounterfactual):
    """Glacier: Gradient-based counterfactual explanation."""
    
    def __init__(self, model, data_name=None, step_size=0.1, max_iter=100, l1_lambda=0.1):
        super().__init__(model)
        self.data_name = data_name
        self.step_size = step_size
        self.max_iter = max_iter
        self.l1_lambda = l1_lambda
        
    def _compute_loss(self, x_tensor, target_class, original_x):
        """Compute loss combining target class probability and sparsity."""
        output = self.model(x_tensor)
        target_prob = output[0, target_class]
        
        # L1 regularization for sparsity
        l1_loss = self.l1_lambda * torch.sum(torch.abs(x_tensor - torch.tensor(original_x).float()))
        
        return -target_prob + l1_loss  # Negative because we want to maximize probability
        
    def generate(self, x, target_class=None, X_train=None, y_train=None):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = get_target_class(self.model, x, y_train)
            
        # Initialize counterfactual
        cf = x.copy()
        best_cf = None
        best_loss = float('inf')
        
        # Gradient descent optimization
        for _ in range(self.max_iter):
            cf_tensor = torch.tensor(cf, requires_grad=True).float()
            loss = self._compute_loss(cf_tensor, target_class, x)
            
            if loss.item() < best_loss and self._is_valid_cf(cf, x, target_class):
                best_cf = cf.copy()
                best_loss = loss.item()
                
            grad = torch.autograd.grad(loss, cf_tensor)[0]
            cf = cf - self.step_size * grad.detach().numpy()
            
            # Project to valid range if needed
            cf = np.clip(cf, x.min(), x.max())
            
        return best_cf.reshape(1, best_cf.shape[0], best_cf.shape[1]) if best_cf is not None else None

    # def generate_batch(self, X_test, y_test, X_train, y_train, selected_indices):
    #     """Generate counterfactuals for multiple instances."""
    #     cfs = []
    #     suc_indices = []
    #     metrics = initialize_metrics()

    #     for i in selected_indices:
    #         try:
    #             start_time = time.time()
    #             x = X_test[i]
    #             target_class = get_target_class(self, x, y_test[i])
    #             cf = self.generate(x, target_class, X_train, y_train)
                
    #             if cf is not None:
    #                 suc_indices.append(i)
    #                 cfs.append(cf)
    #                 update_metrics(self, metrics, x, cf, target_class, start_time)
                    
    #         except Exception as e:
    #             print(f"Error processing instance {i}: {str(e)}")

    #     if self.data_name and suc_indices:
    #         save_metrics_to_csv(self.data_name, metrics, 'AB_CF', 
    #                           [y_test[i] for i in suc_indices])

    #     return np.vstack(cfs) if cfs else None, suc_indices, metrics
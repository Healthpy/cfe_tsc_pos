import numpy as np
import torch
import matplotlib.pyplot as plt
from dtaidistance import dtw
from evaluation.eval import robustness
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import time

class BaseCounterfactual:
    """Base class for counterfactual methods."""
    
    def __init__(self, model, data_name=None):
        self.model = model
        self.data_name = data_name
        self.X_train = None
        self.y_train = None
        
    def set_train_data(self, X_train, y_train):
        """Store training data for counterfactual generation."""
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, x):
        """Make predictions."""
        if isinstance(x, np.ndarray) and len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
            
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                x_tensor = torch.tensor(x).float()
                return np.argmax(self.model(x_tensor).detach().cpu().numpy(), axis=1)[0]
        return self.model.predict(x)[0]
    
    def predict_proba(self, x):
        """Predict probabilities."""
        if isinstance(x, np.ndarray) and len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
            
        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                x_tensor = torch.tensor(x).float()
                return self.model(x_tensor).detach().cpu().numpy()
        return self.model.predict_proba(x)
    
    def _is_valid_cf(self, cf, original_x=None, target_class=None, true_label=None):
        """Validate counterfactual."""
        # Get predictions
        cf_pred = self.predict(cf)
        original_pred = self.predict(original_x)
        
        if cf_pred == original_pred:
            return False
            
        if target_class is not None and target_class == original_pred:
            return False
            
        if true_label is not None and true_label == cf_pred:
            return False
            
        return cf_pred == target_class if target_class is not None else True

    def evaluate(self, x, cf):
        """Evaluate counterfactual quality."""
        robust_results = robustness(self.model, cf, target_class=None)
        
        if len(x.shape) == 3: x = x.reshape(x.shape[1], x.shape[2])
        if len(cf.shape) == 3: cf = cf.reshape(cf.shape[1], cf.shape[2])

        return {
            'valid': self._is_valid_cf(cf, x),
            'conf_orig': robust_results['original_confidence'],
            'gauss_stab': robust_results['gaussian']['prediction_stable'],
            'conf_gauss': robust_results['gaussian']['confidence'],
            'gauss_dist': robust_results['gaussian']['l1_distance'],
            'adv_stab': robust_results['adversarial']['prediction_stable'],
            'conf_adv': robust_results['adversarial']['confidence'],
            'adv_dist': robust_results['adversarial']['l1_distance'],
            'gauss_lip': robust_results['robustness']['gaussian']['lipschitz'],
            'gauss_max_sens': robust_results['robustness']['gaussian']['max_sensitivity'],
            'gauss_avg_sens': robust_results['robustness']['gaussian']['avg_sensitivity'],
            'adv_lip': robust_results['robustness']['adversarial']['lipschitz'],
            'adv_max_sens': robust_results['robustness']['adversarial']['max_sensitivity'],
            'adv_avg_sens': robust_results['robustness']['adversarial']['avg_sensitivity'],
            'sparsity': np.mean(np.isclose(x, cf, rtol=1e-3, atol=1e-3)), #np.mean(np.abs(x - cf) < 1e-6),
            'l1': np.sum(np.abs(x - cf)),
            'l2': np.sqrt(np.sum((x - cf) ** 2)),
            'dtw': dtw.distance(x.flatten(), cf.flatten()) if x.shape == cf.shape else np.inf
        }
    
    def plot(self, x, cf, feature_names=None, title=None):
        """Plot original vs counterfactual."""
        # Shape handling
        if len(x.shape) == 3: x = x.reshape(x.shape[1], x.shape[2])
        if len(cf.shape) == 3: cf = cf.reshape(cf.shape[1], cf.shape[2])
        
        n_features = x.shape[0]
        feature_names = feature_names or [f'Feature {i+1}' for i in range(n_features)]
        
        fig, axs = plt.subplots(n_features, 1, figsize=(10, 2*n_features), sharex=True)
        if n_features == 1: axs = [axs]
        
        for i in range(n_features):
            axs[i].plot(x[i], label='Original', color='blue')
            axs[i].plot(cf[i], label='Counterfactual', color='red', linestyle='--')
            axs[i].set_title(feature_names[i])
            axs[i].legend()
            
        plt.suptitle(title or f'Original vs Counterfactual')
        plt.tight_layout()
        return fig


    def generate_batch(self, X_test, y_test, X_train, y_train, selected_indices):
        """Base implementation of batch generation."""
        cfs = []
        labels = []
        suc_indices = []
        failed_indices = []
        metrics = initialize_metrics()
        total_samples = len(selected_indices)

        print(f"\nProcessing {total_samples} instances...")
        
        for i in selected_indices:
            start_time = time.time()
            x = X_test[i]
            true_label = y_test[i]
            target_class = get_target_class(self, x, true_label)
            
            cf = self.generate(x, target_class, X_train, y_train)
            
            if cf is not None and self._is_valid_cf(cf, x, target_class, true_label):
                suc_indices.append(i)
                cfs.append(cf)
                labels.append(target_class)
                update_metrics(self, metrics, x, cf, target_class, start_time)
            else:
                failed_indices.append(i)

        # Save metrics if successful generations exist
        if self.data_name and suc_indices:
            save_metrics_to_csv(self.data_name, metrics, self.__class__.__name__, 
                              [y_test[i] for i in suc_indices])
        
        return np.vstack(cfs) if cfs else None, np.vstack(labels), suc_indices, metrics

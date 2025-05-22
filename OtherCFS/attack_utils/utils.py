import numpy as np
import tensorflow as tf
import torch
import torchattacks
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

def apply_gaussian_perturbation(x, eps, X_utils=None):
    """Apply Gaussian noise perturbation."""
        # Compute std for each sample, then take the mean
    stdv = np.std(X_utils, axis=0)
    print(f"stdv: {stdv}")
    std_dev = np.mean(np.std(X_utils, axis=0))
    noise = np.random.normal(0, eps*std_dev, x.shape)
    return x + noise

def apply_adversarial_perturbation(x, model, eps=0.1, framework="PY", attack_type="fgsm"):
    """Apply adversarial attack using either FGSM or PGD."""
    if framework == "PY":
        # Ensure CPU device for consistency
        device = torch.device('cpu')
        
        # Convert input to tensor and move to device
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(0)
        x_tensor = x_tensor.to(device)
        
        # Initialize attack
        if attack_type.lower() == "pgd":
            atk = torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=10)
        else:  # default to FGSM
            atk = torchattacks.FGSM(model, eps=eps)
        
        # Manually set device and mode
        atk.device = device
        model.to(device)
        model.eval()
        
        try:
            # Get target from original prediction
            outputs = model(x_tensor)
            target = outputs.argmax(dim=1)
            
            # Run attack
            x_adv = atk(x_tensor, target)
            return x_adv.cpu().detach().numpy()
            
        except Exception as e:
            print(f"Attack failed: {str(e)}")
            return x
        
    elif framework == "TF":
        # Ensure input is a tensor
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        
        if len(x_tensor.shape) == 2:
            x_tensor = tf.expand_dims(x_tensor, axis=0)
        
        # Convert to numpy for compatibility with CleverHans
        x_np = x_tensor.numpy()
        
        # Apply FGSM attack
        x_adv = fast_gradient_method(model, x_np, eps=eps, norm=np.inf)
        
        return tf.convert_to_tensor(x_adv, dtype=tf.float32).numpy()
    else:
        return x
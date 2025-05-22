import numpy as np
import tensorflow as tf
import torch
import torchattacks
# from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

def apply_gaussian_perturbation(x, eps, X_utils=None):
    """Apply Gaussian noise perturbation."""
    # Compute std for each sample, then take the mean
    std_dev = np.mean(np.std(X_utils, axis=0))
    print(f"std_dev: {std_dev}")
    # Generate Gaussian noise
    noise = np.random.normal(0, eps*std_dev, x.shape)
    return x + noise

def apply_adversarial_perturbation(model, x, eps=None, X_utils=None, framework="pytorch", attack_type="fgsm"):
    """Apply adversarial attack using either FGSM or PGD."""
    std_dev = np.mean(np.std(X_utils, axis=0))
    eps = eps*std_dev

    if framework == "pytorch":
        # Ensure CPU device for consistency
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
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
        atk.model.to(device)
        
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
    else:
        return x
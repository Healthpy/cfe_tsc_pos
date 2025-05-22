import torch
import numpy as np
import tensorflow as tf
from utils.pertubation_alg import apply_adversarial_perturbation, apply_gaussian_perturbation
# import quantus

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def compute_local_lipschitz(model, x, eps=0.1, num_samples=100, perturbation="gaussian", 
                           framework="pytorch", attack_type="fgsm"):
    """Compute local Lipschitz estimate for specified perturbation type."""
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    
    x_pred = model.predict_proba(x)
    
    if perturbation == "gaussian":
        # Generate Gaussian perturbations
        perturbed_samples = [apply_gaussian_perturbation(x, eps) for _ in range(num_samples)]
    else:  # adversarial
        # Generate adversarial perturbations
        perturbed = apply_adversarial_perturbation(model, x, eps, framework, attack_type)
        perturbed_samples = [perturbed]
    
    L_values = []
    # Compute Lipschitz constant
    for p_sample in perturbed_samples:
        if np.all(p_sample == x):
            continue
        p_pred = model.predict_proba(p_sample)
        output_diff = np.linalg.norm(p_pred - x_pred)
        input_diff = np.linalg.norm(p_sample - x)

        if input_diff > 0:
            L_values.append(output_diff / input_diff)
    
    return float(max(L_values)) if L_values else 0.0


def compute_sensitivity(model, x, num_samples=20, eps=0.1, framework="pytorch", attack_type="fgsm"):
    """Compute both Gaussian and adversarial sensitivity metrics."""
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    
    orig_pred = model.predict_proba(x)
    sensitivities = {
        'gaussian': [],
        'adversarial': []
    }
    
    # Compute Gaussian sensitivities
    for _ in range(num_samples):
        gaus_pert = apply_gaussian_perturbation(x, eps)
        gaus_pred = model.predict_proba(gaus_pert)
        gaus_sensitivity = np.linalg.norm(gaus_pred - orig_pred)
        sensitivities['gaussian'].append(gaus_sensitivity)
    
        # Compute Adversarial sensitivities
        adv_pert = apply_adversarial_perturbation(model, x, eps, framework, attack_type)
        adv_pred = model.predict_proba(adv_pert)
        adv_sensitivity = np.linalg.norm(adv_pred - orig_pred)
        sensitivities['adversarial'].append(adv_sensitivity)
    
    return {
        'gauss_avg_sens': float(np.mean(sensitivities['gaussian'])),
        'gauss_max_sens': float(np.max(sensitivities['gaussian'])),
        'adv_avg_sens': float(np.mean(sensitivities['adversarial'])),
        'adv_max_sens': float(np.max(sensitivities['adversarial']))
    }

def compute_robustness_metrics(model, x, eps=0.1, framework="pytorch", attack_type="fgsm"):
    """Compute robustness metrics for both Gaussian and adversarial perturbations."""
    try:
        # Get sensitivities for both perturbation types
        sensitivity_metrics = compute_sensitivity(model, x, eps=eps, framework=framework, 
                                               attack_type=attack_type)
        
        # Compute Lipschitz constants for both perturbation types
        gaussian_lipschitz = compute_local_lipschitz(model, x, eps=eps, perturbation="gaussian")
        adversarial_lipschitz = compute_local_lipschitz(model, x, eps=eps, perturbation="adversarial",
                                                       framework=framework, attack_type=attack_type)
        
        # Compute metrics
        return {
            "gaussian": {
                "lipschitz": gaussian_lipschitz,
                "max_sensitivity": sensitivity_metrics['gauss_max_sens'],
                "avg_sensitivity": sensitivity_metrics['gauss_avg_sens']
            },
            "adversarial": {
                "lipschitz": adversarial_lipschitz,
                "max_sensitivity": sensitivity_metrics['adv_max_sens'],
                "avg_sensitivity": sensitivity_metrics['adv_avg_sens'],
                "attack_type": attack_type
            }
        }
        
    except Exception as e:
        print(f"Error in computing metrics: {str(e)}")
        return get_default_robustness_metrics()

def robustness(model, cf, target_class, std_dev=0.1, eps=0.1):
    """Evaluate robustness using multiple metrics."""
    try:
        # Get original prediction and confidence
        orig_metrics = get_basic_robustness(model, cf, target_class, std_dev, eps)
        
        # Add direct robustness metrics
        orig_metrics["robustness"] = compute_robustness_metrics(model, cf, eps)
        
        return orig_metrics
        
    except Exception as e:
        print(f"Error in robustness evaluation: {str(e)}")
        # return get_default_metrics()

def get_basic_robustness(model, cf, target_class, std_dev=0.1, eps=0.1, framework="pytorch", attack_type="fgsm"):
    """Evaluate robustness using gaussian and one type of adversarial attack."""
    # try:
    if len(cf.shape) == 2:
        cf = cf.reshape(1, cf.shape[0], cf.shape[1])
    
    original_pred = model.predict(cf)[0]
    pos, original_conf = confidence(model, cf)
    print(f"pos: {pos}, original_conf: {original_conf}")
    
    gaussian_cf = apply_gaussian_perturbation(cf, std_dev)
    adv_cf = apply_adversarial_perturbation(model, cf, eps, framework, attack_type)
    
    gaussian_pred = model.predict(gaussian_cf)[0]
    adv_pred = model.predict(adv_cf)[0]
    
    return {
        "original_confidence": original_conf,
        "gaussian": {
            "prediction_stable": original_pred == gaussian_pred,
            "confidence": confidence(model, gaussian_cf, position=pos),
            "l1_distance": float(np.mean(np.abs(cf - gaussian_cf)))
        },
        "adversarial": {
            "prediction_stable": original_pred == adv_pred,
            "confidence": confidence(model, adv_cf, position=pos),
            "l1_distance": float(np.mean(np.abs(cf - adv_cf))),
            "attack_type": attack_type
        }
    }
    # except Exception as e:
    #     print(f"Error in robustness evaluation: {str(e)}")
    #     return get_default_metrics()

def get_default_metrics():
    """Return default values for metrics."""
    return {
        "original_confidence": 0.0,
        "gaussian": {
            "prediction_stable": False,
            "confidence": 0.0,
            "l1_distance": 0.0
        },
        "fgsm": {  # Changed from 'adversarial' to 'fgsm'
            "prediction_stable": False,
            "confidence": 0.0,
            "l1_distance": 0.0
        },
        "pgd": {
            "prediction_stable": False,
            "confidence": 0.0,
            "l1_distance": 0.0
        }
    }

def get_default_robustness_metrics():
    """Return default values for robustness metrics."""
    metric_structure = {
        "lipschitz": 0.0,
        "max_sensitivity": 0.0,
        "avg_sensitivity": 0.0
    }
    return {
        "gaussian": metric_structure.copy(),
        "adversarial": {**metric_structure.copy(), "attack_type": "none"}
    }

def confidence(model, cf, position=None):
    """Compute model's confidence (probability) for the predicted class."""
    probs = model.predict_proba(cf)
    position = np.argmax(probs) if position is None else position
    return position, probs[0][position]
    
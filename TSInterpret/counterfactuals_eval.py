import sys
import os
from attack_utils.utils import apply_adversarial_perturbation, apply_gaussian_perturbation
# Path to your local package directory
local_package_path = os.path.abspath("/Users/emmanuelchukwu/Library/CloudStorage/OneDrive-TUEindhoven/Desktop/XAI/TSCE_Tool/JHoelli/TSInterpret")
# local_package_path = os.path.abspath(r"C:\Users\20235732\OneDrive - TU Eindhoven\Desktop\XAI\TSCE_Tool\JHoelli\TSInterpret\TSInterpret")
if local_package_path not in sys.path:
    sys.path.insert(0, local_package_path)

import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from dtaidistance import dtw
from TSInterpret.InterpretabilityModels.counterfactual.COMTECF import COMTECF
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF import NativeGuideCF
from TSInterpret.InterpretabilityModels.counterfactual.SETSCF import SETSCF
from sklearn.metrics.pairwise import euclidean_distances
from tslearn.datasets import UCR_UEA_datasets
# Custom imports
from ClassificationModels.CNN_T import ResNetBaseline, UCRDataset, fit, get_all_preds
from ClassificationModes.LSTMFCN import LSTMFCN
import sklearn
from tensorflow.keras.models import load_model as tf_load_model # type: ignore
import argparse
from ClassificationModels.LSTM_T import LSTM



# Set random seeds for reproducibility
SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['TF_DETERMINISTIC_OPS'] = '1'
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except:
    pass

def get_model(model_name, n_features, n_classes, model_path, max_seq_len=None, framework='PY'):
    """Get model by name with framework specification."""
    if framework == 'TF':
        models = {'resnet': lambda: tf_load_model(model_path + '_tf_best_model.hdf5'),
            }
    else:
        models = {
            'lstmfcn': lambda: LSTMFCN(n_features=n_features, 
                                      n_classes=n_classes, 
                                      max_seq_len=max_seq_len),
            'lstm': lambda: LSTM(input_size=n_features, 
                               hidden_size=128, 
                               num_classes=n_classes),
            'resnet': lambda: ResNetBaseline(in_channels=n_features, 
                                           num_pred_classes=n_classes)
        }
    return models[model_name.lower()]()

def get_cf_methods(method_names, model, train_x, train_y, y_train, framework='PY'):
    """Get counterfactual methods by name."""
    available_methods = {
        "comte": lambda: COMTECF(model, (train_x,train_y), backend='PYT', mode='feat', method='opt'),
        "tsevo": lambda: TSEvo(model=model, data=(train_x, train_y), mode='feat', backend='PYT', epochs=200),
        "ng_cf": lambda: NativeGuideCF(model, (train_x, train_y), backend='PYT', mode='feat', method='NUN_CF'),
    }
    
    # Handle SETS separately as it requires TensorFlow
    if 'sets' in method_names and framework == 'TF':
        available_methods["sets"] = lambda: SETSCF(model,
                                                 (train_x, y_train),
                                                 backend='TF',
                                                 mode='feat',
                                                 min_shapelet_len=3,
                                                 max_shapelet_len=20,
                                                 time_contract_in_mins_per_dim=1,
                                                 fit_shapelets=False)
    
    selected_methods = {}
    for name in method_names:
        if name.lower() in available_methods:
            selected_methods[name.upper()] = available_methods[name.lower()]()
        elif name.lower() == 'sets' and framework != 'TF':
            print("Warning: SETS method requires TensorFlow framework. Skipping...")
    
    return selected_methods

def plot_counterfactual(x, cf, dataset, method, index, orig_label, cf_label):
    """Plot original and counterfactual time series."""
    plt.figure(figsize=(10, 5))

    # Ensure arrays are 3D (samples, features, timesteps)
    if len(x.shape) == 2:
        x = x.reshape(1, -1, x.shape[-1])
    if len(cf.shape) == 2:
        cf = cf.reshape(1, -1, cf.shape[-1])

    # Get number of features/channels
    n_features = x.shape[1]
    
    if n_features == 1:
        # Univariate case
        plt.plot(x[0, 0, :], 'b-', label=f'Original (class {orig_label})', alpha=0.7)
        plt.plot(cf[0, 0, :], 'r--', label=f'Counterfactual (class {cf_label})', alpha=0.7)
        plt.legend()
    else:
        # Multivariate case
        for i in range(n_features):
            plt.subplot(n_features, 1, i+1)
            plt.plot(x[0, i, :], 'b-', label=f'Original (class {orig_label})' if i == 0 else '', alpha=0.7)
            plt.plot(cf[0, i, :], 'r--', label=f'Counterfactual (class {cf_label})' if i == 0 else '', alpha=0.7)
            plt.ylabel(f'Feature {i+1}')
            if i == 0:
                plt.legend()
    
    plt.suptitle(f'{dataset} - {method} - Sample {index}')
    
    # Create plot directory if it doesn't exist
    plot_dir = f"plots/{dataset}"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.tight_layout()
    # Save plot
    plt.savefig(f"{plot_dir}/{method}_instance_{index}.png")
    plt.close()

def plot_all_variants(x, cf, dataset, X_utils, method, index, orig_label, cf_label, model, get_predictions, args, eps_values):
    """Plot original, counterfactual and their variants at different epsilon levels."""
    plt.figure(figsize=(15, 8))
    
    variants = {eps: {
        'orig_gauss': apply_gaussian_perturbation(x, X_utils=X_utils, eps=eps),
        'orig_adv': apply_adversarial_perturbation(x, model, X_utils=X_utils, framework=args.framework, eps=eps),
        'cf_gauss': apply_gaussian_perturbation(cf.numpy() if not isinstance(cf, np.ndarray) else cf, X_utils=X_utils, eps=eps),
        'cf_adv': apply_adversarial_perturbation(cf.numpy() if not isinstance(cf, np.ndarray) else cf, model, X_utils=X_utils, framework=args.framework, eps=eps)
    } for eps in eps_values}

    # Plot original and its variants
    plt.subplot(2, 1, 1)
    plt.plot(x.flatten(), 'b-', label=f'Original (class {orig_label})', linewidth=2)

    # Plot Gaussian variants of original
    for eps in eps_values:
        plt.plot(variants[eps]['orig_gauss'].flatten(), 'b:', alpha=0.3, 
                label=f'Orig Gaussian ε={eps:.2f}')

    # Plot adversarial variants of original
    for eps in eps_values:
        plt.plot(variants[eps]['orig_adv'].flatten(), 'g--', alpha=0.3,
                label=f'Orig Adversarial ε={eps:.2f}')

    # Ensure unique legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'Original Sample and Variants - {dataset} - {method} - Sample {index}')
    plt.grid(True, alpha=0.3)

    # Plot counterfactual and its variants
    plt.subplot(2, 1, 2)
    plt.plot(cf.flatten(), 'r-', label=f'Counterfactual (class {cf_label})', linewidth=2)

    # Plot Gaussian variants of counterfactual
    for eps in eps_values:
        plt.plot(variants[eps]['cf_gauss'].flatten(), 'r:', alpha=0.3, 
                label=f'CF Gaussian ε={eps:.2f}')

    # Plot adversarial variants of counterfactual
    for eps in eps_values:
        plt.plot(variants[eps]['cf_adv'].flatten(), 'm--', alpha=0.3,
                label=f'CF Adversarial ε={eps:.2f}')

    # Ensure unique legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'Counterfactual and Variants - {dataset} - {method} - Sample {index}')
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Create plot directory if it doesn't exist
    plot_dir = f"plots/{dataset}/variant_comparison"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{method}_instance_{index}_variants.png", bbox_inches='tight')
    plt.close()

def load_model(model_path, model_name, model, framework='PY'):
    """Load trained model based on framework."""
    if framework == 'TF':
        try:
            loaded_model = tf_load_model(model_path + '_tf_best_model.hdf5')
            print(f"Loaded TF model from {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Error loading TF model: {e}")
            return None
    else:
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded PyTorch model from {model_path}")
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            return None

def plot_epsilon_comparison(results, method_name, dataset):
    """Plot comparison of robustness and confidence across different epsilon values."""
    eps_keys = sorted(results[method_name]['robustness_gaussian_by_eps'].keys())
    eps_values = [float(eps) for eps in eps_keys]
    
    # Get mean values for each epsilon
    orig_rob_gauss = [np.mean(results[method_name]['original_robustness_gaussian_by_eps'][eps]) for eps in eps_keys]
    orig_conf_gauss = [np.mean(results[method_name]['original_confidence_gaussian_by_eps'][eps]) for eps in eps_keys]
    orig_rob_adv = [np.mean(results[method_name]['original_robustness_adv_by_eps'][eps]) for eps in eps_keys]
    orig_conf_adv = [np.mean(results[method_name]['original_confidence_adv_by_eps'][eps]) for eps in eps_keys]
    
    cf_rob_gauss = [np.mean(results[method_name]['robustness_gaussian_by_eps'][eps]) for eps in eps_keys]
    cf_conf_gauss = [np.mean(results[method_name]['confidence_gaussian_by_eps'][eps]) for eps in eps_keys]
    cf_rob_adv = [np.mean(results[method_name]['robustness_adv_by_eps'][eps]) for eps in eps_keys]
    cf_conf_adv = [np.mean(results[method_name]['confidence_adv_by_eps'][eps]) for eps in eps_keys]
    
    plt.figure(figsize=(15, 10))
    
    # Plot Gaussian noise effects
    plt.subplot(2, 1, 1)
    plt.plot(eps_values, orig_rob_gauss, 'b-', label='Original Robustness', marker='o')
    plt.plot(eps_values, orig_conf_gauss, 'b--', label='Original Confidence', marker='s')
    plt.plot(eps_values, cf_rob_gauss, 'r-', label='CF Robustness', marker='o')
    plt.plot(eps_values, cf_conf_gauss, 'r--', label='CF Confidence', marker='s')
    plt.title(f'Gaussian Noise Effects - {dataset} - {method_name}')
    plt.xlabel('Epsilon')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    # Plot Adversarial effects
    plt.subplot(2, 1, 2)
    plt.plot(eps_values, orig_rob_adv, 'b-', label='Original Robustness', marker='o')
    plt.plot(eps_values, orig_conf_adv, 'b--', label='Original Confidence', marker='s')
    plt.plot(eps_values, cf_rob_adv, 'r-', label='CF Robustness', marker='o')
    plt.plot(eps_values, cf_conf_adv, 'r--', label='CF Confidence', marker='s')
    plt.title(f'Adversarial Effects - {dataset} - {method_name}')
    plt.xlabel('Epsilon')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Create plot directory if it doesn't exist
    plot_dir = f"plots/{dataset}/epsilon_effects"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{method_name}_epsilon_effects.png")
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate counterfactuals for time series classification models')
    parser.add_argument('--models', nargs='+', default=['lstmfcn'],
                       help='Models to evaluate: lstmfcn, lstm, resnet')
    parser.add_argument('--datasets', nargs='+', default=['ECG200', 'ECG5000', 'Epilepsy', 'TwoLeadECG'],
                        choices=['ECG200', 'ECG5000', 'Epilepsy', 'TwoLeadECG', 'CardiacArrhythmia'],
                       help='Datasets to use')
    parser.add_argument('--cf_methods', nargs='+', default=['ng_cf', 'comte', 'tsevo'],
                       help='Counterfactual methods to use: comte, tsevo, ng_cf, sets')
    parser.add_argument('--framework', default='PY', choices=['PY', 'TF'],
                       help='Framework to use for counterfactual methods: PY (PyTorch), TF (TensorFlow)')
    args = parser.parse_args()

    all_results = []
    
    for dataset in args.datasets:

        print(f"\nProcessing dataset: {dataset}")
        dataset_individual_results = []
        
        # Load dataset
        X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
        # Preprocess data
        train_x = np.swapaxes(X_train, -1, -2)
        test_x = np.swapaxes(X_test, -1, -2)

        enc1 = sklearn.preprocessing.OneHotEncoder(sparse=False)
        train_y = enc1.fit_transform(y_train.reshape(-1,1))
        test_y = enc1.transform(y_test.reshape(-1,1))

        for model_name in args.models:
            print(f"\nEvaluating {model_name.upper()} on {dataset}")
            
            # Initialize model with framework specification
            model = get_model(model_name, 
                             n_features=train_x.shape[1],
                             n_classes=train_y.shape[1],
                             max_seq_len=train_x.shape[2],
                            model_path=f"models/{dataset}_{model_name.upper()}",
                             framework=args.framework)
            
            # Load trained model with updated path handling
            if args.framework == 'TF':
                model_path = f"models/{dataset}_{model_name.upper()}" 
            else:
                model_path = f"models/{dataset}_{model_name.upper()}_best.pth"
            
            model = load_model(model_path, model_name, model, framework=args.framework)
            if model is None:
                print(f"Could not load model for {dataset} and {model_name}, skipping...")
                continue

            # Update prediction handling for TF models
            if args.framework == 'TF':
                def get_predictions(x):
                    return model(x, training=False).numpy()
            else:
                def get_predictions(x):
                    return torch.nn.functional.softmax(model(torch.from_numpy(x).float())).detach().numpy()
            
            # Create data loaders
            train_dataset = UCRDataset(train_x.astype(np.float64), train_y.astype(np.int64))
            test_dataset = UCRDataset(test_x.astype(np.float64), test_y.astype(np.int64))
            # Define batch size
            batch_size = 32
            # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

            # y_pred, labels = get_all_preds(model, test_loader)

            # Define counterfactual methods
            methods = get_cf_methods(args.cf_methods, model, train_x, train_y, y_train, framework=args.framework)
            if not methods:
                print(f"No valid counterfactual methods selected for {model_name} on {dataset}")
                continue

            # Metrics storage
            results = {method: {
                "success_rate": 0,
                # "sparsity": [],
                "tp_sparsity": [],	
                "seg_sparsity": [], 
                "l1": [],
                "l2": [],
                "dtw": [],
                "confidence": [], 
                "robustness_gaussian": [],
                "robustness_adv": [],
                "confidence_gaussian": [],
                "confidence_adv": [],
                "generation_time": [],
                "original_confidence": [],
                "original_robustness_gaussian": [],
                "original_confidence_gaussian": [],
                "original_robustness_adv": [],
                "original_confidence_adv": [],
                "robustness_gaussian_by_eps": {},
                "confidence_gaussian_by_eps": {},
                "robustness_adv_by_eps": {},
                "confidence_adv_by_eps": {},
                "original_robustness_gaussian_by_eps": {},
                "original_confidence_gaussian_by_eps": {},
                "original_robustness_adv_by_eps": {},
                "original_confidence_adv_by_eps": {},
            } for method in methods}

            # Select sample indices before method loop
            num_samples = min(100, len(test_x))
            # np.random.seed(42)  # Ensure same samples are selected each time
            sample_indices = np.random.choice(len(test_x), num_samples, replace=False)
            print(f"Selected samples: {sample_indices}")

            for method_name, method in methods.items():
                print(f"\nProcessing method: {method_name}")
                success_count = 0
                for i in sample_indices:
                    try:
                        x = test_x[i].reshape(1,test_x.shape[1],-1)
                        _x = torch.from_numpy(x).float()
                        y_pred = get_predictions(x)
                        
                        if np.argmax(y_pred) == np.argmax(test_y[i]):
                            original_label = test_y[i]
                            
                            # Calculate metrics for original sample
                            orig_conf = y_pred.max()
                            results[method_name]["original_confidence"].append(orig_conf)
                            orig_class = np.argmax(y_pred)

                            # Start timing
                            start_time = time.time()
                            if method_name == "COMTE":
                                cf, cf_label = method.explain(x)
                                cf_label = cf_label[0]
                            elif method_name == "TSEVO":
                                cf, cf_label = method.explain(x, np.argmax(y_pred, axis=1))
                                cf_label = np.argmax(cf_label)
                            elif method_name == "NG_CF":
                                cf, cf_label = method.explain(x, np.argmax(y_pred, axis=1)[0])
                            elif method_name == "SETS":
                                method.fit(occlusion_threshhold=1e-1,remove_multiclass_shapelets=True)
                                cf, cf_label = method.explain(test_x[i], target=None)
                            else:
                                print(f"Unknown method: {method_name}")
                                continue
                            
                            # End timing
                            generation_time = time.time() - start_time
                            
                            print(f"Counterfactual generated in {generation_time:.3f} seconds")
                        else:
                            print(f"Skipping sample {i} due to label mismatch.")
                            continue
                    except Exception as e:
                        print(f"Error generating counterfactual for sample {i} using {method_name}: {str(e)}")
                    # try:
                    if cf is not None:
                        success_count += 1
                        results[method_name]["generation_time"].append(generation_time)
                        
                        # Calculate sparsity at two levels
                        # 1. Time point level sparsity
                        tp_sparsity = np.mean(np.isclose(x, cf, rtol=1e-3, atol=1e-3))
                        
                        # 2. Segment level sparsity
                        signal_length = x.shape[-1]
                        segment_size = max(int(0.1 * signal_length), 1)  # 10% of signal length, minimum 1
                        
                        # Reshape to segments and calculate mean for each segment
                        n_segments = signal_length // segment_size
                        x_segments = x[..., :n_segments*segment_size].reshape(*x.shape[:-1], n_segments, segment_size)
                        cf_segments = cf[..., :n_segments*segment_size].reshape(*cf.shape[:-1], n_segments, segment_size)
                        
                        # Calculate segment means
                        x_segment_means = np.mean(x_segments, axis=-1)
                        cf_segment_means = np.mean(cf_segments, axis=-1)
                        
                        # Compare segments
                        segment_sparsity = np.mean(np.isclose(x_segment_means, cf_segment_means, rtol=1e-3, atol=1e-3))
                        
                        # # Average both sparsity measures
                        # sparsity = (timepoint_sparsity + segment_sparsity) / 2
        
                        results[method_name]["tp_sparsity"].append(tp_sparsity)
                        results[method_name]["seg_sparsity"].append(segment_sparsity)
                        
                        # L1 distance
                        l1_dist = np.sum(np.abs(x - cf))
                        results[method_name]["l1"].append(l1_dist)
                        
                        # L2 distance
                        l2_dist = np.sqrt(np.sum((x - cf) ** 2))
                        results[method_name]["l2"].append(l2_dist)
                        
                        # DTW distance
                        dtw_dist = dtw.distance(x.flatten(), cf.flatten())
                        results[method_name]["dtw"].append(dtw_dist)
                        # Confidence of counterfactual
                        if args.framework == 'TF':
                            conf = y_pred[0].max()
                        else:
                            cf_input = cf if isinstance(cf, np.ndarray) else cf.numpy()
                            if len(cf_input.shape) == 2:
                                cf_input = np.expand_dims(cf_input, axis=0)
                            cf_input = torch.from_numpy(cf_input).float()
                            conf = torch.nn.functional.softmax(model(cf_input)).detach().numpy().max()
                            print(f"Confidence: {conf}")
                        
                        results[method_name]["confidence"].append(conf)
                        
                        # Store original prediction class
                        cf_pred_class = get_predictions(cf.numpy() if not isinstance(cf, np.ndarray) else cf)[0].argmax()
                        print(f"CF Predicted class: {cf_pred_class}")
                        X_utils = X_test
                        
                        epses = np.linspace(0.0, 2.0, 11)
                        for eps in epses:
                            eps_key = f"{eps:.2f}"
                            
                            # Initialize lists for this epsilon if not exists
                            for key in ['robustness_gaussian_by_eps', 'confidence_gaussian_by_eps',
                                      'robustness_adv_by_eps', 'confidence_adv_by_eps',
                                      'original_robustness_gaussian_by_eps', 'original_confidence_gaussian_by_eps',
                                      'original_robustness_adv_by_eps', 'original_confidence_adv_by_eps']:
                                if eps_key not in results[method_name][key]:
                                    results[method_name][key][eps_key] = []
                            
                            # Test original sample with current epssilon
                            noisy_orig = apply_gaussian_perturbation(x, X_utils=X_utils, eps=eps)
                            noisy_orig_pred_probs = get_predictions(noisy_orig)
                            orig_robustness_gaussian = float(np.argmax(noisy_orig_pred_probs) == orig_class)
                            results[method_name]["original_robustness_gaussian_by_eps"][eps_key].append(orig_robustness_gaussian)
                            results[method_name]["original_confidence_gaussian_by_eps"][eps_key].append(noisy_orig_pred_probs[0][orig_class])

                            adv_orig = apply_adversarial_perturbation(x, model, X_utils=X_utils, framework=args.framework, eps=eps)
                            adv_orig_pred_probs = get_predictions(adv_orig)
                            orig_robustness_adv = float(np.argmax(adv_orig_pred_probs) == orig_class)
                            results[method_name]["original_robustness_adv_by_eps"][eps_key].append(orig_robustness_adv)
                            results[method_name]["original_confidence_adv_by_eps"][eps_key].append(adv_orig_pred_probs[0][orig_class])

                            # Test counterfactual with current epsilon
                            noisy_cf = apply_gaussian_perturbation(cf.numpy() if not isinstance(cf, np.ndarray) else cf, X_utils=X_utils, eps=eps)
                            if args.framework == 'TF':
                                if len(noisy_cf.shape) == 2:
                                    noisy_cf = np.expand_dims(noisy_cf, axis=0)
                                noisy_pred_probs = get_predictions(noisy_cf)
                            else:
                                _noisy_cf = torch.from_numpy(noisy_cf).float()
                                if len(_noisy_cf.shape) == 2:
                                    _noisy_cf = _noisy_cf.unsqueeze(0)
                                noisy_pred_probs = get_predictions(_noisy_cf.numpy())
                            
                            noisy_pred = noisy_pred_probs.argmax()
                            print(f"CF Noisy class: {noisy_pred}")
                            print(f"CF Predicted class: {cf_pred_class}")
                            print(f"probs:{noisy_pred_probs[0]}")
                            noisy_conf = noisy_pred_probs[0][cf_pred_class]
                            robustness_gaussian = (noisy_pred == cf_pred_class)
                            results[method_name]["robustness_gaussian_by_eps"][eps_key].append(robustness_gaussian)
                            results[method_name]["confidence_gaussian_by_eps"][eps_key].append(noisy_conf)

                            adv_cf = apply_adversarial_perturbation(cf.numpy() if not isinstance(cf, np.ndarray) else cf, model, X_utils=X_utils, framework=args.framework, eps=eps)
                            if args.framework == 'TF':
                                adv_pred_probs = get_predictions(adv_cf)
                            else:
                                _adv_cf = torch.from_numpy(adv_cf).float()
                                if len(_adv_cf.shape) == 2:
                                    _adv_cf = _adv_cf.unsqueeze(0)
                                adv_pred_probs = get_predictions(_adv_cf.numpy())
                            
                            adv_pred = adv_pred_probs.argmax()
                            adv_conf = adv_pred_probs[0][cf_pred_class]
                            robustness_adv = (adv_pred == cf_pred_class)
                            results[method_name]["robustness_adv_by_eps"][eps_key].append(robustness_adv)
                            results[method_name]["confidence_adv_by_eps"][eps_key].append(adv_conf)
                            
                            # Plot and save counterfactual
                            # orig_class = np.argmax(y_pred, axis=1)[0]
                            plot_counterfactual(x, cf, dataset, method_name, i, orig_class, cf_label)
                        
                        # After generating epsilon variants, add:
                        plot_all_variants(x, cf, dataset, X_utils, method_name, i, orig_class, cf_label, 
                                       model, get_predictions, args, [float(eps_key) for eps_key in results[method_name]['robustness_gaussian_by_eps'].keys()])
                        
                        # Store individual result with dataset-specific list
                        individual_result = {
                            'Method': method_name,
                            'Model': model_name.upper(),
                            'Sample_Index': i,
                            'Original_Class': orig_class,
                            'Counterfactual_Class': cf_label,
                            'Generation_Time': round(generation_time, 4),
                            'Tp_Sparsity': round(tp_sparsity, 4),
                            'Seg_Sparsity': round(segment_sparsity, 4),
                            'DTW': round(dtw_dist, 4),
                            'L1': round(l1_dist, 4),
                            'L2': round(l2_dist, 4), 
                            'CF_Conf': round(conf, 4),
                        }
                        
                        # Add epsilon-specific metrics for individual results
                        for eps_key in results[method_name]['robustness_gaussian_by_eps'].keys():
                            individual_result.update({
                                f'Orig_Rob_Gauss_eps{eps_key}': round(float(results[method_name]["original_robustness_gaussian_by_eps"][eps_key][-1]), 4),
                                f'Orig_Conf_Gauss_eps{eps_key}': round(float(results[method_name]["original_confidence_gaussian_by_eps"][eps_key][-1]), 4),
                                f'Orig_Rob_Adv_eps{eps_key}': round(float(results[method_name]["original_robustness_adv_by_eps"][eps_key][-1]), 4),
                                f'Orig_Conf_Adv_eps{eps_key}': round(float(results[method_name]["original_confidence_adv_by_eps"][eps_key][-1]), 4),
                                f'CF_Rob_Gauss_eps{eps_key}': round(float(results[method_name]["robustness_gaussian_by_eps"][eps_key][-1]), 4),
                                f'CF_Conf_Gauss_eps{eps_key}': round(float(results[method_name]["confidence_gaussian_by_eps"][eps_key][-1]), 4),
                                f'CF_Rob_Adv_eps{eps_key}': round(float(results[method_name]["robustness_adv_by_eps"][eps_key][-1]), 4),
                                f'CF_Conf_Adv_eps{eps_key}': round(float(results[method_name]["confidence_adv_by_eps"][eps_key][-1]), 4),
                            })
                        
                        dataset_individual_results.append(individual_result)
                        
                        # Save individual results progressively and create pivot
                        ind_df = pd.DataFrame(dataset_individual_results)
                        
                        # Save both original and pivoted results
                        base_path = "results"
                        method_path = f"{base_path}/{method_name}"
                        os.makedirs(method_path, exist_ok=True)
                        ind_df.to_csv(f"{method_path}/{dataset}_cf_results.csv", index=False)
                    

                results[method_name]["success_rate"] = round(success_count / num_samples, 4)
                
                # Plot epsilon comparison
                plot_epsilon_comparison(results, method_name, dataset)
                
                # Store results with dataset information
                summary_result = {
                    'Dataset': dataset,
                    'Method': method_name,
                    'Validity': f"{np.mean(results[method_name]['success_rate']):.2f}",
                    'Gen_Time': f"{np.mean(results[method_name]['generation_time']):.3f} ± {np.std(results[method_name]['generation_time']):.2f}",
                    'TP_Sparsity': f"{np.mean(results[method_name]['tp_sparsity']):.3f} ± {np.std(results[method_name]['tp_sparsity']):.2f}",
                    'Seg_Sparsity': f"{np.mean(results[method_name]['seg_sparsity']):.3f} ± {np.std(results[method_name]['seg_sparsity']):.2f}",
                    'L1': f"{np.mean(results[method_name]['l1']):.3f} ± {np.std(results[method_name]['l1']):.2f}",
                    'L2': f"{np.mean(results[method_name]['l2']):.3f} ± {np.std(results[method_name]['l2']):.2f}",
                    'DTW': f"{np.mean(results[method_name]['dtw']):.3f} ± {np.std(results[method_name]['dtw']):.2f}",
                    'CF_Confidence': f"{np.mean(results[method_name]['confidence']):.3f} ± {np.std(results[method_name]['confidence']):.2f}",
                }

                # Add epsilon-specific metrics for summary results
                for eps_key in results[method_name]['robustness_gaussian_by_eps'].keys():
                    summary_result.update({
                        f'Orig_Rob_Gauss_eps{eps_key}': f"{np.mean(results[method_name]['original_robustness_gaussian_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['original_robustness_gaussian_by_eps'][eps_key]):.2f}",
                        f'Orig_Conf_Gauss_eps{eps_key}': f"{np.mean(results[method_name]['original_confidence_gaussian_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['original_confidence_gaussian_by_eps'][eps_key]):.2f}",
                        f'Orig_Rob_Adv_eps{eps_key}': f"{np.mean(results[method_name]['original_robustness_adv_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['original_robustness_adv_by_eps'][eps_key]):.2f}",
                        f'Orig_Conf_Adv_eps{eps_key}': f"{np.mean(results[method_name]['original_confidence_adv_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['original_confidence_adv_by_eps'][eps_key]):.2f}",
                        f'CF_Rob_Gauss_eps{eps_key}': f"{np.mean(results[method_name]['robustness_gaussian_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['robustness_gaussian_by_eps'][eps_key]):.2f}",
                        f'CF_Conf_Gauss_eps{eps_key}': f"{np.mean(results[method_name]['confidence_gaussian_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['confidence_gaussian_by_eps'][eps_key]):.2f}",
                        f'CF_Rob_Adv_eps{eps_key}': f"{np.mean(results[method_name]['robustness_adv_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['robustness_adv_by_eps'][eps_key]):.2f}",
                        f'CF_Conf_Adv_eps{eps_key}': f"{np.mean(results[method_name]['confidence_adv_by_eps'][eps_key]):.3f} ± {np.std(results[method_name]['confidence_adv_by_eps'][eps_key]):.2f}",
                    })

                all_results.append(summary_result)

                # Load or create summary file
                summary_file = 'results/all_datasets_and_methods_summary.csv'
                if os.path.exists(summary_file):
                    summary_df = pd.read_csv(summary_file)
                    mask = ~((summary_df['Dataset'] == dataset) & 
                            (summary_df['Method'] == method_name))
                    summary_df = summary_df[mask]
                else:
                    summary_df = pd.DataFrame()
                
                # Add new results and save
                new_results_df = pd.DataFrame(all_results)
                summary_df = pd.concat([summary_df, new_results_df], ignore_index=True)
                summary_df = summary_df.sort_values(by=['Dataset', 'Method'])

                
                summary_df.to_csv(summary_file, index=False, float_format='%.4f')
                print(f"Summary results saved to {summary_file}")

if __name__ == "__main__":
    main()
import argparse
from explainers import METHOD_MAP
from train_models.tscnn import ModelWrapper, create_model
import numpy as np
from data.load_datasets import load_ucr_dataset, load_custom_dataset
from utils.utils import (initialize_metrics, save_metrics_to_csv, 
                        plot_example_counterfactual, extract_metrics, 
                        get_target_class, plot_counterfactual)
from utils.pertubation_alg import apply_adversarial_perturbation, apply_gaussian_perturbation
import os
import pandas as pd
import time
from dtaidistance import dtw
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run counterfactual generation with specified settings')
    
    # Model settings
    parser.add_argument('--framework', type=str, default='pytorch',
                      choices=['pytorch', 'tensorflow'],
                      help='Deep learning framework to use')
    
    parser.add_argument('--model_type', type=str, default='lstmfcn',
                      choices=['lstmfcn', 'cnn', 'bayesian'],
                      help='Type of model to use')
    
    parser.add_argument('--datasets', nargs='+', default=['ECG200', 'ECG5000', 'TwoLeadECG', 'Epilepsy'],
                      help='List of datasets to process')
    
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    
    parser.add_argument('--methods', nargs='+', default=['AB_CF'],
                      choices=list(METHOD_MAP.keys()),
                      help='Counterfactual methods to use')
    
    parser.add_argument('--model_name', type=str, default=None,
                      help='Specific name for the model (optional)')
    
    # Add custom data arguments
    parser.add_argument('--custom_data', type=str, default=None,
                      help='Path to custom dataset file')
    parser.add_argument('--feature_cols', nargs='+', default=None,
                      help='Names of feature columns for custom data')
    parser.add_argument('--label_col', type=str, default=None,
                      help='Name of label column for custom data')
    parser.add_argument('--time_col', type=str, default=None,
                      help='Name of time column for custom data')
    parser.add_argument('--id_col', type=str, default=None,
                      help='Name of ID column for custom data')
    
    args = parser.parse_args()
    
    # Set default values if using custom data
    if args.custom_data:
        if args.feature_cols is None:
            args.feature_cols = []
        if args.label_col is None:
            args.label_col = 'label'
        if args.time_col is None:
            args.time_col = 'timestamp'
        if args.id_col is None:
            args.id_col = 'id'
            
    return args

def compute_robustness_metrics(model, x, cf, X_utils, framework="pytorch"):
    """Compute robustness metrics for both original sample and counterfactual."""
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    if len(cf.shape) == 2:
        cf = cf.reshape(1, cf.shape[0], cf.shape[1])
    
    # Original predictions
    x_pred = model.predict_proba(x)
    x_class = np.argmax(x_pred)
    cf_pred = model.predict_proba(cf)
    cf_class = np.argmax(cf_pred)
    
    results = {}
    # Test with different epsilon values
    epses = np.linspace(0.0, 2, 11)
    for eps in epses:
        eps_key = f"{eps:.2f}"
        
        # Original sample robustness
        noisy_x = apply_gaussian_perturbation(x, eps, X_utils)
        noisy_x_pred = model.predict_proba(noisy_x)
        adv_x = apply_adversarial_perturbation(model, x, eps, X_utils, framework)
        adv_x_pred = model.predict_proba(adv_x)
        
        # Counterfactual robustness
        noisy_cf = apply_gaussian_perturbation(cf, eps, X_utils)
        noisy_cf_pred = model.predict_proba(noisy_cf)
        adv_cf = apply_adversarial_perturbation(model, cf, eps, X_utils, framework)
        adv_cf_pred = model.predict_proba(adv_cf)
        
        results[eps_key] = {
            "original": {
                "gaussian_stable": float(np.argmax(noisy_x_pred) == x_class),
                "gaussian_confidence": float(noisy_x_pred[0][x_class]),
                "adversarial_stable": float(np.argmax(adv_x_pred) == x_class),
                "adversarial_confidence": float(adv_x_pred[0][x_class])
            },
            "counterfactual": {
                "gaussian_stable": float(np.argmax(noisy_cf_pred) == cf_class),
                "gaussian_confidence": float(noisy_cf_pred[0][cf_class]),
                "adversarial_stable": float(np.argmax(adv_cf_pred) == cf_class),
                "adversarial_confidence": float(adv_cf_pred[0][cf_class])
            }
        }
    
    return results

def compute_sparsity_metrics(x, cf):
    """Compute sparsity metrics at timepoint and segment levels."""
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    if len(cf.shape) == 2:
        cf = cf.reshape(1, cf.shape[0], cf.shape[1])
        
    # Timepoint sparsity
    tp_sparsity = np.mean(np.isclose(x, cf, rtol=1e-3, atol=1e-3))
    
    # Segment sparsity
    signal_length = x.shape[-1]
    segment_size = max(int(0.1 * signal_length), 1)
    n_segments = signal_length // segment_size
    
    x_segments = x[..., :n_segments*segment_size].reshape(*x.shape[:-1], n_segments, segment_size)
    cf_segments = cf[..., :n_segments*segment_size].reshape(*cf.shape[:-1], n_segments, segment_size)
    
    x_segment_means = np.mean(x_segments, axis=-1)
    cf_segment_means = np.mean(cf_segments, axis=-1)
    
    segment_sparsity = np.mean(np.isclose(x_segment_means, cf_segment_means, rtol=1e-3, atol=1e-3))
    
    return tp_sparsity, segment_sparsity

def compute_distance_metrics(x, cf):
    """Compute L1, L2, and DTW distances."""
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
    if len(cf.shape) == 2:
        cf = cf.reshape(1, cf.shape[0], cf.shape[1])
        
    l1_dist = float(np.sum(np.abs(x - cf)))
    l2_dist = float(np.sqrt(np.sum((x - cf) ** 2)))
    dtw_dist = float(dtw.distance(x.flatten(), cf.flatten())) if x.shape == cf.shape else float('inf')
    
    return l1_dist, l2_dist, dtw_dist

def plot_epsilon_comparison(results_df, method_name, dataset):
    """Plot comparison of robustness and confidence across different epsilon values."""
    # Get all epsilon values from column names
    eps_values = np.linspace(0.0, 2, 11)
    method_results = results_df[results_df['Method'] == method_name]
    
    # Calculate mean values for each epsilon
    orig_rob_gauss = [method_results[f'Orig_Rob_Gauss_eps{eps:.2f}'].mean() for eps in eps_values]
    orig_conf_gauss = [method_results[f'Orig_Conf_Gauss_eps{eps:.2f}'].mean() for eps in eps_values]
    orig_rob_adv = [method_results[f'Orig_Rob_Adv_eps{eps:.2f}'].mean() for eps in eps_values]
    orig_conf_adv = [method_results[f'Orig_Conf_Adv_eps{eps:.2f}'].mean() for eps in eps_values]
    
    cf_rob_gauss = [method_results[f'CF_Rob_Gauss_eps{eps:.2f}'].mean() for eps in eps_values]
    cf_conf_gauss = [method_results[f'CF_Conf_Gauss_eps{eps:.2f}'].mean() for eps in eps_values]
    cf_rob_adv = [method_results[f'CF_Rob_Adv_eps{eps:.2f}'].mean() for eps in eps_values]
    cf_conf_adv = [method_results[f'CF_Conf_Adv_eps{eps:.2f}'].mean() for eps in eps_values]
    
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

def generate_and_evaluate_counterfactuals(dataset_name, args):
    """Generate and evaluate counterfactuals with specified settings."""
    if args.custom_data:
        X_train, y_train, X_test, y_test = load_custom_dataset(
            args.custom_data,
            feature_columns=args.feature_cols,
            label_column=args.label_col,
            time_column=args.time_col,
            id_column=args.id_col
        )
        dataset_name = os.path.splitext(os.path.basename(args.custom_data))[0]
    else:
        X_train, y_train, X_test, y_test = load_ucr_dataset(dataset_name)
    
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    seq_length = X_train.shape[2]
    
    print(f"\nInitializing {args.model_type} model...")
    model = create_model(
        n_features=n_features,
        n_classes=n_classes,
        seq_length=seq_length,
        framework=args.framework,
        model_type=args.model_type
    )
    
    model_name = f"{dataset_name}_{args.framework}_{args.model_type}"
    model.train_model(X_train, y_train, model_name, epochs=args.epochs)
    print(f"Model accuracy: {model.accuracy_score(X_test, y_test):.4f}")
    
    n_test_samples = min(100, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test_samples, replace=False)
    
    dataset_individual_results = []
    all_results = []
    
    for method_name in args.methods:
        print(f"\nProcessing method: {method_name}")
        
        method = METHOD_MAP[method_name](model, data_name=dataset_name)
        # Set training data for methods that need it
        if hasattr(method, 'set_train_data'):
            method.set_train_data(X_train, y_train)
        success_count = 0
        
        for i in test_indices:
            try:
                x = X_test[i].reshape(1, X_test.shape[1], -1)
                y_true = y_test[i]
                X_utils = X_test
                if np.argmax(model.predict_proba(x)) == y_true:
                    start_time = time.time()
                    
                    # Get target class
                    target_class = get_target_class(model, x, y_true)
                    cf, cf_label = method.explain(x, target_class)
                    generation_time = time.time() - start_time
                    
                    if cf is not None:
                        print(f"Counterfactual found for sample {i}: {cf_label}")
                        success_count += 1
                        
                        # Plot counterfactual example
                        plot_counterfactual(x, cf, dataset_name, method_name, i, 
                                        np.argmax(model.predict_proba(x)), cf_label)
                        
                        # Compute all metrics
                        tp_sparsity, seg_sparsity = compute_sparsity_metrics(x, cf)
                        l1_dist, l2_dist, dtw_dist = compute_distance_metrics(x, cf)
                        robustness_results = compute_robustness_metrics(model, x, cf, X_utils, framework=args.framework)
                        
                        # Store individual result
                        result = {
                            'Method': method_name,
                            'Dataset': dataset_name,
                            'Sample_Index': i,
                            'Original_Class': np.argmax(model.predict_proba(x)),
                            'Counterfactual_Class': cf_label,
                            'Generation_Time': round(generation_time, 4),
                            'Tp_Sparsity': round(tp_sparsity, 4),
                            'Seg_Sparsity': round(seg_sparsity, 4),
                            'L1': round(l1_dist, 4),
                            'L2': round(l2_dist, 4),
                            'DTW': round(dtw_dist, 4)
                        }
                        
                        # Add epsilon-specific metrics
                        for eps_key, eps_results in robustness_results.items():
                            result.update({
                                f'Orig_Rob_Gauss_eps{eps_key}': eps_results['original']['gaussian_stable'],
                                f'Orig_Conf_Gauss_eps{eps_key}': eps_results['original']['gaussian_confidence'],
                                f'Orig_Rob_Adv_eps{eps_key}': eps_results['original']['adversarial_stable'],
                                f'Orig_Conf_Adv_eps{eps_key}': eps_results['original']['adversarial_confidence'],
                                f'CF_Rob_Gauss_eps{eps_key}': eps_results['counterfactual']['gaussian_stable'],
                                f'CF_Conf_Gauss_eps{eps_key}': eps_results['counterfactual']['gaussian_confidence'],
                                f'CF_Rob_Adv_eps{eps_key}': eps_results['counterfactual']['adversarial_stable'],
                                f'CF_Conf_Adv_eps{eps_key}': eps_results['counterfactual']['adversarial_confidence']
                            })
                        
                        dataset_individual_results.append(result)
                        
                        # Save results progressively
                        results_dir = os.path.join("results", method_name)
                        os.makedirs(results_dir, exist_ok=True)
                        pd.DataFrame(dataset_individual_results).to_csv(
                            os.path.join(results_dir, f"{dataset_name}_cf_results.csv"),
                            index=False
                        )
                    else:
                        print(f"Counterfactual generation failed for sample {i}")
                else:
                    print(f"Sample {i} is misclassified, skipping...")
                    continue
                    
            except Exception as e:
                print(f"Error with {method_name} on sample {i}: {str(e)}")
                continue

        # Plot epsilon comparison
        if dataset_individual_results:
            results_df = pd.DataFrame(dataset_individual_results)
            plot_epsilon_comparison(results_df, method_name, dataset_name)
        
        # Compute method summary with confidence intervals
        if success_count > 0:
            summary = {
                'Dataset': dataset_name,
                'Method': method_name,
                'Validity': round(success_count / n_test_samples, 4)
            }
            
            # Calculate mean and std for metrics
            results_df = pd.DataFrame(dataset_individual_results)
            method_results = results_df[results_df['Method'] == method_name]
            
            for metric in ['Generation_Time', 'Tp_Sparsity', 'Seg_Sparsity', 'L1', 'L2', 'DTW']:
                summary[metric] = f"{method_results[metric].mean():.3f} ± {method_results[metric].std():.2f}"
            
            # Add epsilon-specific metrics
            eps_columns = [col for col in method_results.columns if col.startswith(('Orig_Rob_', 'Orig_Conf_', 'CF_Rob_', 'CF_Conf_'))]
            for col in eps_columns:
                summary[col] = f"{method_results[col].mean():.3f} ± {method_results[col].std():.2f}"
            
            all_results.append(summary)

    
    # Save summary results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv('results/all_datasets_and_methods_summary.csv', 
                         mode='a', header=not os.path.exists('results/all_datasets_and_methods_summary.csv'),
                         index=False)
    
    return all_results

if __name__ == "__main__":
    args = parse_args()
    all_dataset_results = []
    
    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        dataset_results = generate_and_evaluate_counterfactuals(dataset_name, args)
        all_dataset_results.extend(dataset_results)
    
    print("\nProcessing complete. Results saved in the 'results' directory.")


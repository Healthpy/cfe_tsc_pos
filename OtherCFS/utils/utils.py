import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# Create directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)


import time
import numpy as np
import pandas as pd
import os

def initialize_metrics():
    """Initialize metrics dictionary with abbreviated names."""
    return {
        'target': [], 'times': [], 'valid': [],
        'conf_orig': [], 'conf_gauss': [], 'conf_adv': [],
        'gauss_stab': [], 'gauss_dist': [],
        'adv_stab': [], 'adv_dist': [],
        'sparsity': [], 'l1': [], 'l2': [], 'dtw': [],
        'gauss_lip': [],
        'gauss_max_sens': [],
        'gauss_avg_sens': [],
        'adv_lip': [],
        'adv_max_sens': [],
        'adv_avg_sens': [],
    }

def get_target_class(model, x, y_true):
    """Get target class for counterfactual generation."""
    # Ensure x has correct shape for prediction
    if len(x.shape) == 2:
        x = x.reshape(1, x.shape[0], x.shape[1])
        
    # Get current prediction
    pred = model.predict_proba(x)[0]  # Get first sample's predictions
    current_class = np.argmax(pred)
    
    # If current class is the same as true class, find another target
    if y_true is not None and current_class == y_true:
        # Get second highest probability class
        sorted_classes = np.argsort(pred)[::-1]
        target_class = sorted_classes[1]  # Second highest probability class
    else:
        # Use most probable class as target
        target_class = current_class
        
    return int(target_class)

def update_metrics(model, metrics, x, cf, target_class, start_time):
    """Update metrics dictionary with evaluation results."""
    # Update basic timing and target info
    metrics['times'].append(time.time() - start_time)
    metrics['target'].append(target_class)
    
    # Get evaluation results
    eval_metrics = model.evaluate(x, cf)
    
    # Update all metrics including nested robustness metrics
    for key in metrics.keys():
        if key in eval_metrics:
            value = eval_metrics[key]
            if isinstance(value, tuple):
                value = value[1]
            metrics[key].append(float(value))
            
        elif 'gaussian_' in key and key.replace('gaussian_', '') in eval_metrics.get('robustness', {}).get('gaussian', {}):
            nested_key = key.replace('gaussian_', '')
            metrics[key].append(float(eval_metrics['robustness']['gaussian'][nested_key]))
        elif 'adversarial_' in key and key.replace('adversarial_', '') in eval_metrics.get('robustness', {}).get('adversarial', {}):
            nested_key = key.replace('adversarial_', '')
            metrics[key].append(float(eval_metrics['robustness']['adversarial'][nested_key]))


def compute_metrics(counterfactuals, success_indices, original_data, cf_method, n_test_samples):
    method_name = cf_method.__class__.__name__
    metrics_dict = {}

    if len(counterfactuals) > 0:
        success_rate = len(success_indices) / n_test_samples
        print(f"{method_name} success rate: {success_rate:.2f}")

        metric_names = [
            'validity', 'original_confidence',
            'gaussian_stable', 'gaussian_confidence', 'gaussian_distance',
            'adversarial_stable', 'adversarial_confidence', 'adversarial_distance',
            'sparsity', 'l1_distance', 'l2_distance', 'dtw_distance'
        ]
        metric_values = {metric: [] for metric in metric_names}

        # Compute metrics
        for i, idx in enumerate(success_indices):
            metrics = cf_method.evaluate(original_data[idx], counterfactuals[i])
            for metric in metric_names:
                metric_values[metric].append(metrics.get(metric, np.nan))

        # Compute averages and print summary
        for metric in metric_names:
            if len(metric_values[metric]) > 0:
                avg_value = np.nanmean(metric_values[metric])
                metrics_dict[metric] = avg_value
                if metric in ['gaussian_stable', 'adversarial_stable']:
                    print(f"{method_name} {metric}: {avg_value:.2%}")
                else:
                    print(f"{method_name} average {metric}: {avg_value:.4f}")
            else:
                metrics_dict[metric] = None

    return metrics_dict

def save_metrics_to_csv(data_name, metrics, method_name, ori_labels):
    """Save computed metrics to a CSV file inside the 'results' folder with proper column format."""
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    filename = f"results/{method_name}_{data_name}_metrics.csv"
    
    # Ensure all arrays have the same length
    n_samples = len(ori_labels)
    processed_metrics = {}
    
    for key, values in metrics.items():
        if isinstance(values, list):
            # Round numeric values to 4 decimal places
            if all(isinstance(v, (int, float)) for v in values):
                values = [round(float(v), 4) for v in values]
                
            # Pad or truncate lists to match n_samples
            if len(values) > n_samples:
                processed_metrics[key] = values[:n_samples]
            else:
                # Pad with zeros or appropriate default values
                processed_metrics[key] = values + [0] * (n_samples - len(values))
        else:
            # If not a list, round single value and create list
            if isinstance(values, (int, float)):
                values = round(float(values), 4)
            processed_metrics[key] = [values] * n_samples
    
    # Convert metrics dictionary to DataFrame
    df = pd.DataFrame(processed_metrics)
    
    # Add original labels to the DataFrame
    df.insert(0, "original_label", ori_labels)
    
    # Round all numeric columns to 4 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    # Save DataFrame to CSV with specified float format
    df.to_csv(filename, index=False, float_format='%.4f')

    print(f"Metrics saved to {filename}")

def plot_example_counterfactual(cf_method, x, cf, example_idx, y_true, cf_label, feature_names=None):
    """
    Plot and save an example counterfactual using the method's built-in plot function.
    """
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Create a new figure with a specific size
    plt.figure(figsize=(12, 6))

    # Ensure x and cf are 2D
    if len(x.shape) == 3:
        x = x.reshape(x.shape[1], x.shape[2])
    if len(cf.shape) == 3:
        cf = cf.reshape(cf.shape[1], cf.shape[2])
            
    n_features = x.shape[0]
    time_steps = x.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
    # Create subplots for each feature
    fig, axs = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True)
    
    # Handle the case where there's only one feature
    if n_features == 1:
        axs = [axs]
        
    # Plot each feature
    for i in range(n_features):
        axs[i].plot(x[i], 'b-', label=f'Original (class {y_true})', alpha=0.7)
        axs[i].plot(cf[i], 'r--', label=f'Counterfactual (class {cf_label[0]})', alpha=0.7)
        # axs[i].set_title(f'{feature_names[i]}')
        axs[i].legend()
        axs[i].set_ylabel(f'Feature {i+1}')
        # axs[i].grid(True)
        
    # Add overall title
    dataset_name = cf_method.data_name if hasattr(cf_method, 'data_name') else 'Unknown'
    plt.suptitle(f"Counterfactual for {dataset_name} - Sample {example_idx}", fontsize=14, y=0.95)
            
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot with dataset name in the filename
    plot_filename = f"plots/{cf_method.__class__.__name__}_{dataset_name}_{example_idx}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    # Close the figure to free memory
    plt.close('all')

    print(f"Counterfactual plot saved to {plot_filename}")


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

def extract_metrics(counterfactuals, success_indices, metrics, n_samples, method_name, dataset_name):
    """Extract metrics including Quantus metrics."""
    # Calculate true validity rate (number of valid CFs over total samples)
    n_valid = sum(1 for valid in metrics['valid'] if valid)
    validity_rate = n_valid / n_samples if n_samples > 0 else 0
    
    # Calculate current method's metrics
    results = {
        'method': method_name,
        'dataset': dataset_name,
        'success_rate': round(len(success_indices) / n_samples if metrics['times'] else 0, 4),
        'avg_time': round(np.mean(metrics['times']) if metrics['times'] else 0, 4),
        'valid_rate': round(validity_rate, 4),
        'sparsity': round(np.mean(metrics['sparsity']) if metrics['sparsity'] else 0, 4),
        'l1': round(np.mean(metrics['l1']) if metrics['l1'] else 0, 4),
        'l2': round(np.mean(metrics['l2']) if metrics['l2'] else 0, 4),
        'dtw': round(np.mean(metrics['dtw']) if metrics['dtw'] else 0, 4),
        'conf_orig': round(np.mean(metrics['conf_orig']) if metrics['conf_orig'] else 0, 4),	
        'gauss_stab': round(np.mean(metrics['gauss_stab']) if metrics.get('gauss_stab') else 0, 4),
        'conf_gauss': round(np.mean(metrics['conf_gauss']) if metrics.get('conf_gauss') else 0, 4),
        'adv_stab': round(np.mean(metrics['adv_stab']) if metrics.get('adv_stab') else 0, 4),
        'conf_adv': round(np.mean(metrics['conf_adv']) if metrics.get('conf_adv') else 0, 4),
        'gauss_lip': round(np.mean(metrics['gauss_lip']) if metrics.get('gauss_lip') else 0, 4),
        'gauss_max_sens': round(np.mean(metrics['gauss_max_sens']) if metrics.get('gauss_max_sens') else 0, 4),
        'gauss_avg_sens': round(np.mean(metrics['gauss_avg_sens']) if metrics.get('gauss_avg_sens') else 0, 4),
        'adv_lip': round(np.mean(metrics['adv_lip']) if metrics.get('adv_lip') else 0, 4),
        'adv_max_sens': round(np.mean(metrics['adv_max_sens']) if metrics.get('adv_max_sens') else 0, 4),
        'adv_avg_sens': round(np.mean(metrics['adv_avg_sens']) if metrics.get('adv_avg_sens') else 0, 4),
    }
    
    # Load or create summary file
    summary_file = 'results/all_datasets_summary.csv'
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        mask = ~((summary_df['dataset'] == dataset_name) & 
                (summary_df['method'] == method_name))
        summary_df = summary_df[mask]
    else:
        summary_df = pd.DataFrame()
    
    # Add new results and save
    new_row = pd.DataFrame([results])
    summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    summary_df = summary_df.sort_values(['dataset', 'method'])
    
    # Round all numeric columns to 4 decimal places
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    summary_df.to_csv(summary_file, index=False, float_format='%.4f')
    
    # Print update confirmation with rounded values
    print(f"\nUpdated summary for {method_name} on {dataset_name}")
    print(f"Success rate: {results['success_rate']:.3f}")
    print(f"Gaussian stability: {results['gauss_stab']:.3f}")
    print(f"Adversarial stability: {results['adv_stab']:.3f}")
    
    return results


import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_results(train_losses, val_losses, val_metrics, representation, config):
    """
    Plot training and validation losses for a specific rotation representation.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        val_metrics (list): List of dictionaries containing metrics per epoch
        representation (str): Name of the rotation representation
        config: Configuration object with plot_dir attribute
    """
    if not config.save_plots:
        return
        
    # Create results directory if it doesn't exist
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{representation} - Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(config.plot_dir, f'{representation}_loss.png'))
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(val_metrics) + 1)
    
    metrics_to_plot = ['l1_distance', 'l2_distance', 'chordal_distance', 'geodesic_distance']
    
    for i, metric_name in enumerate(metrics_to_plot):
        values = [metrics[metric_name] for metrics in val_metrics]
        plt.subplot(2, 2, i+1)
        plt.plot(epochs, values)
        plt.title(f'{metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, f'{representation}_metrics.png'))
    plt.close()

def plot_comparison(results_dict, config):
    """
    Plot comparison of all rotation representations for each metric.
    
    Args:
        results_dict (dict): Dictionary with keys as representation names and values as
                            dictionaries containing train_losses, val_losses, and val_metrics
        config: Configuration object with plot_dir attribute
    """
    if not config.save_plots:
        return
        
    # Create results directory if it doesn't exist
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Plot comparison of losses
    plt.figure(figsize=(12, 8))
    for rep, results in results_dict.items():
        plt.plot(results['train_losses'], label=f'{rep} Train')
        plt.plot(results['val_losses'], label=f'{rep} Val')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(config.plot_dir, 'comparison_loss.png'))
    plt.close()
    
    # Plot comparison of metrics
    metrics_to_plot = ['l1_distance', 'l2_distance', 'chordal_distance', 'geodesic_distance']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        for rep, results in results_dict.items():
            values = [metrics[metric] for metrics in results['val_metrics']]
            plt.plot(values, label=rep)
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f'Comparison of {metric}')
        plt.legend()
        plt.savefig(os.path.join(config.plot_dir, f'comparison_{metric}.png'))
        plt.close()

def plot_final_comparison(results_dict, config):
    """
    Plot final comparison of all metrics for all representations.
    
    Args:
        results_dict (dict): Dictionary with keys as representation names and values as
                            dictionaries containing train_losses, val_losses, and val_metrics
        config: Configuration object with plot_dir attribute
    """
    if not config.save_plots:
        return
        
    # Create results directory if it doesn't exist
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Get the final metrics for each representation
    final_metrics = {}
    for rep, results in results_dict.items():
        final_metrics[rep] = results['val_metrics'][-1]  # Get the last epoch's metrics
    
    # Plot final comparison for each metric
    metrics_to_plot = ['l1_distance', 'l2_distance', 'chordal_distance', 'geodesic_distance']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        
        reps = list(final_metrics.keys())
        values = [final_metrics[rep][metric] for rep in reps]
        
        plt.bar(reps, values)
        plt.title(f'Final {metric}')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, 'final_comparison.png'))
    plt.close()

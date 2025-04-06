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
    
    metrics_to_plot = ['l1', 'l2', 'chordal', 'geodesic']
    
    for i, metric_name in enumerate(metrics_to_plot):
        values = [metrics[metric_name].item() for metrics in val_metrics]
        plt.subplot(2, 2, i+1)
        plt.plot(epochs, values)
        plt.title(f'{metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.plot_dir, f'{representation}_metrics.png'))
    plt.close()

def plot_box_metrics(results_dict, config):
    # Create results directory if it doesn't exist
    os.makedirs(config.plot_dir, exist_ok=True)
    
    rep_names = list(results_dict.keys())  # e.g. 4 representations
    metric_names = ['l1', 'l2', 'chordal', 'geodesic']
    n_reps = len(rep_names)
    group_gap = n_reps + 1  # gap between groups
    
    data_to_plot = []  # list to store data for each box
    positions = []     # x-axis positions for each box
    group_centers = [] # for labeling groups
    
    # Iterate over each error metric
    for m_index, m in enumerate(metric_names):
        for r_index, rep in enumerate(rep_names):
            # For each representation, extract the values of the metric from all epochs
            values = [metrics[m].item() for metrics in results_dict[rep]['val_metrics']]
            data_to_plot.append(values)
            # Calculate position for the box
            pos = m_index * group_gap + r_index
            positions.append(pos)
        # Calculate the center of the current group for the x-axis tick label
        center = m_index * group_gap + (n_reps - 1) / 2.0
        group_centers.append(center)
    
    plt.figure(figsize=(12, 8))
    bp = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    # Set colors for boxes (assign one color per representation)
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % n_reps])
    
    plt.xlabel('Error Metric')
    plt.ylabel('Error')
    plt.title('Box Plot of Error Metrics for Each Representation')
    plt.xticks(group_centers, metric_names)
    
    # Create legend manually to indicate which color corresponds to which representation
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[i], label=rep_names[i]) for i in range(n_reps)]
    plt.legend(handles=legend_patches, title='Representation', loc='upper right')
    
    plt.savefig(os.path.join(config.plot_dir, 'box_plot_metrics.png'))
    plt.close()
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_results(train_losses, val_losses, val_metrics, representation, config):     
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
    
    metrics_to_plot = config.metrics
    
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
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Group experiments by training loss function metrics
    # result_dict's key =  "representation_metric"
    # e.g. "euler_l1"
    groups = {}  
    rep_set = set()
    for key, result in results_dict.items():
        rep, loss_metric = key.rsplit('_', 1)
        rep_set.add(rep)
        if loss_metric not in groups:
            groups[loss_metric] = {}
        values = [m[loss_metric].item() for m in result['val_metrics']]
        groups[loss_metric][rep] = values

    # Order groups by config.metrics and representations
    groups_order = config.metrics  
    rep_order = sorted(list(rep_set))
    n_reps = len(rep_order)
    group_gap = n_reps + 1  
    
    data_to_plot = []
    positions = []
    group_centers = []
    
    # Build data_to_plot and positions for boxplot() 
    for i, loss_metric in enumerate(groups_order):
        group_data = groups.get(loss_metric, {})
        for j, rep in enumerate(rep_order):
            if rep in group_data:
                data_to_plot.append(group_data[rep])
            else:
                data_to_plot.append([])
            pos = i * group_gap + j
            positions.append(pos)
        center = i * group_gap + (n_reps - 1) / 2.0
        group_centers.append(center)
    
    plt.figure(figsize=(12, 8))
    bp = plt.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    # Use a consistent color for each representation across groups.
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for i, box in enumerate(bp['boxes']):
        rep_index = i % n_reps  # same order as rep_order
        box.set_facecolor(colors[rep_index])
    
    plt.xlabel('Training Loss Function Metrics')
    plt.ylabel('Error Value')
    plt.title('Validation Error for Each Experiment')
    plt.xticks(group_centers, groups_order)
    
    # Create a legend mapping each color to its representation.
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[i], label=rep_order[i]) for i in range(n_reps)]
    plt.legend(handles=legend_patches, title='Representation', loc='upper right')
    
    plt.savefig(os.path.join(config.plot_dir, 'box_plot_quantitative_metrics.png'))
    plt.close()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from config import Config
from dataset import PointCloudAlignmentDataset
from model import RotationEstimationModel
from trainer import Trainer
from visualization import plot_training_results, plot_box_metrics


def main():
    # Load configuration
    config = Config()

    # Print configuration
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Representations: {list(config.representations.keys())}")
    print(f"Metrics: {config.metrics}")

    # Create directory for plots
    os.makedirs(config.plot_dir, exist_ok=True)

    # Dictionary to store trainers for each representation
    trainers = {}
    
    # Dictionary to store results for comparison plots
    results_dict = {}

    # Train a model for each representation
    for representation, output_size in config.representations.items():
        ## MODIFIED HERE that we need to train with 16 metrics
        for metric in config.metrics:
            print(f"\n{'='*50}")
            print(f"Training model for {representation} representation with {metric} loss")
            print(f"{'='*50}")

            # Create datasets
            train_dataset = PointCloudAlignmentDataset(
                mode="train", device=config.device, representation=representation
            )
            test_dataset = PointCloudAlignmentDataset(
                mode="test", device=config.device, representation=representation
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size, shuffle=False
            )

            # Create model
            model = RotationEstimationModel(config.input_dim, output_size)

            # Create trainer with specific representation and metric
            trainer = Trainer(model, config, representation)
            
            ## MODIFIED HERE TO TRAIN WITH 16 METRICS
            trainer.metrics = metric

            # Train model
            trainer.train(train_loader, test_loader)

            # Store trainer for visualization
            key = f"{representation}_{metric}"
            trainers[key] = trainer
            
            # Store results for comparison plots
            results_dict[key] = {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'val_metrics': trainer.val_metrics
            }

    # Plot results
    print("\nPlotting results...")
    
    # Plot individual results for each representation and metric combination
    for key in trainers.keys():
        print(f"Plotting results for {key}")
        trainer = trainers[key]
        plot_training_results(
            trainer.train_losses, 
            trainer.val_losses, 
            trainer.val_metrics, 
            key, 
            config
        )
    
    # Plot box plot comparison for error metrics (16 boxes: 4 metrics × 4 representations)
    print("Plotting box plot comparison for error metrics...")
    plot_box_metrics(results_dict, config)


if __name__ == "__main__":
    main()

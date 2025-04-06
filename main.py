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
from visualization import *


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

    # Train a model for each representation
    for representation, output_size in config.representations.items():
        print(f"\n{'='*50}")
        print(f"Training model for {representation} representation")
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

        # Create trainer
        trainer = Trainer(model, config, representation)

        # Train model
        trainer.train(train_loader, test_loader)

        # Store trainer for visualization
        trainers[representation] = trainer

    # Plot results
    print("\nPlotting results...")
    # TODO: Plot and save the results
    # Need to plot the results for each representation 4x4 pairs 
    # Plot the results for each representation
    for representation in trainers.keys():
        print(f"Plotting results for {representation} representation")
        trainer = trainers[representation]
        trainer.plot_results()


if __name__ == "__main__":
    main()

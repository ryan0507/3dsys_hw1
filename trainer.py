import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from rotations import compute_metrics


class Trainer:
    def __init__(self, model, config, representation):
        """
        Trainer for rotation estimation models.

        Args:
            model (nn.Module): Model to train
            config (Config): Configuration object
            representation (str): Rotation representation to use
        """
        self.model = model
        self.config = config
        self.representation = representation
        self.device = config.device

        # Move model to device
        self.model = self.model.to(self.device)

        # TODO: Initialize necessary variables
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 학습 히스토리 저장을 위한 변수들
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        # TODO: Implement training loop

        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            # Input and targets to 'device'
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward updates 
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        # TODO: Implement evaluation loop
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Input and targets to 'device'
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update variables
                val_loss += loss.item()
                all_preds.append(outputs)
                all_targets.append(targets)
        
        avg_loss = val_loss / len(val_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, self.representation)
        
        return avg_loss, metrics

    def train(self, train_loader, val_loader):
        """Train the model for multiple epochs."""
        # TODO: Implement training loop
        print(f"Training model for {self.representation} representation")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            
            val_loss, metrics = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            # Save all of the metrics for each epoch box plots
            self.val_metrics.append(metrics)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Metrics: L1={metrics['l1_distance']:.4f}, L2={metrics['l2_distance']:.4f}, " + 
                  f"Chordal={metrics['chordal_distance']:.4f}, Geodesic={metrics['geodesic_distance']:.4f}")
            
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")

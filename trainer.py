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
            # 입력과 타겟을 디바이스로 이동
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Loss 계산
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        # 평균 loss 계산 후 반환
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
                # 입력과 타겟을 디바이스로 이동
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Loss 계산
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # 예측값과 타겟 저장
                all_preds.append(outputs)
                all_targets.append(targets)
        
        # 평균 loss 계산
        avg_loss = val_loss / len(val_loader)
        
        # 모든 예측값과 타겟 합치기
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 메트릭 계산
        metrics = compute_metrics(all_preds, all_targets, self.representation)
        
        return avg_loss, metrics

    def train(self, train_loader, val_loader):
        """Train the model for multiple epochs."""
        # TODO: Implement training loop
        print(f"Training model for {self.representation} representation")
        
        for epoch in range(self.config.epochs):
            # 한 에폭 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증 데이터로 평가
            val_loss, metrics = self.evaluate(val_loader)
            
            # 학습 히스토리 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(metrics)
            
            # 최고 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # 진행 상황 출력
            print(f"Epoch {epoch+1}/{self.config.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Metrics: L1={metrics['l1_distance']:.4f}, L2={metrics['l2_distance']:.4f}, " + 
                  f"Chordal={metrics['chordal_distance']:.4f}, Geodesic={metrics['geodesic_distance']:.4f}")
        
        # 학습 완료 후 최고 모델 로드
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
    
    def plot_results(self):
        """Plot training results."""
        if not self.config.save_plots:
            return
            
        # 결과 저장 디렉토리 생성
        os.makedirs(self.config.plot_dir, exist_ok=True)
        
        # 학습 및 검증 손실 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.representation} - Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.plot_dir, f'{self.representation}_loss.png'))
        plt.close()
        
        # 메트릭 그래프
        plt.figure(figsize=(12, 8))
        epochs = range(1, len(self.val_metrics) + 1)
        
        metrics_to_plot = ['l1_distance', 'l2_distance', 'chordal_distance', 'geodesic_distance']
        
        for i, metric_name in enumerate(metrics_to_plot):
            values = [metrics[metric_name] for metrics in self.val_metrics]
            plt.subplot(2, 2, i+1)
            plt.plot(epochs, values)
            plt.title(f'{metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, f'{self.representation}_metrics.png'))
        plt.close()

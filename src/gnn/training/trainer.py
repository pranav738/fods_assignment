"""
Training pipeline for spatio-temporal GNN models
"""

import os
import time
from typing import Dict, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.metrics import compute_metrics, masked_rmse, masked_mae


class SpatioTemporalTrainer:
    """
    Trainer for spatio-temporal GNN models

    Handles:
    - Training loop with early stopping
    - Validation and testing
    - Checkpointing
    - TensorBoard logging
    - Learning rate scheduling
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 scaler: Optional[object] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 epochs: int = 100,
                 patience: int = 15,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = './checkpoints',
                 log_dir: str = './runs',
                 clip_grad: float = 5.0):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            scaler: Data scaler for inverse transform
            learning_rate: Learning rate
            weight_decay: L2 regularization
            epochs: Maximum number of epochs
            patience: Early stopping patience
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            clip_grad: Gradient clipping value
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.clip_grad = clip_grad

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Loss function
        self.criterion = masked_rmse

        # Logging
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.epochs} [Train]')

        for batch_idx, (x, y, adj) in enumerate(pbar):
            # Move to device
            x = x.to(self.device)
            y = y.to(self.device)
            adj = adj.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x, adj)

            # Compute loss
            loss = self.criterion(predictions, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, data_loader: DataLoader, phase: str = 'val') -> Dict[str, float]:
        """
        Validate model

        Args:
            data_loader: Data loader
            phase: Phase name ('val' or 'test')

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f'Epoch {self.current_epoch+1}/{self.epochs} [{phase}]')

            for x, y, adj in pbar:
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)
                adj = adj.to(self.device)

                # Forward pass
                predictions = self.model(x, adj)

                # Compute loss
                loss = self.criterion(predictions, y)

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions and targets
                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())

                pbar.set_postfix({'loss': loss.item()})

        # Compute average loss
        avg_loss = total_loss / num_batches

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Inverse transform if scaler provided
        if self.scaler is not None:
            all_predictions_np = all_predictions.numpy()
            all_targets_np = all_targets.numpy()

            # Reshape for inverse transform
            shape = all_predictions_np.shape
            all_predictions_np = self.scaler.inverse_transform(
                all_predictions_np.reshape(-1, 1),
                feature_idx=0
            ).reshape(shape)
            all_targets_np = self.scaler.inverse_transform(
                all_targets_np.reshape(-1, 1),
                feature_idx=0
            ).reshape(shape)
        else:
            all_predictions_np = all_predictions.numpy()
            all_targets_np = all_targets.numpy()

        # Compute comprehensive metrics
        metrics = compute_metrics(all_predictions_np, all_targets_np)
        metrics['loss'] = avg_loss

        return metrics

    def train(self) -> Dict[str, any]:
        """
        Full training loop with early stopping

        Returns:
            Training history
        """
        print(f"\nTraining on device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(self.val_loader, phase='val')
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/val_RMSE', val_metrics['RMSE'], epoch)
            self.writer.add_scalar('Metrics/val_MAE', val_metrics['MAE'], epoch)
            self.writer.add_scalar('Metrics/val_MAPE', val_metrics['MAPE'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val RMSE: {val_metrics['RMSE']:.4f}")
            print(f"Val MAE: {val_metrics['MAE']:.4f}")
            print(f"Val MAPE: {val_metrics['MAPE']:.2f}%")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                self.save_checkpoint('best_model.pth')
                print("✓ New best model saved")
            else:
                self.patience_counter += 1
                print(f"Early stopping: {self.patience_counter}/{self.patience}")

                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")

        # Load best model for final evaluation
        self.load_checkpoint('best_model.pth')

        # Final test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.validate(self.test_loader, phase='test')

        print("\n" + "="*50)
        print("FINAL TEST RESULTS")
        print("="*50)
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Test MAE: {test_metrics['MAE']:.4f}")
        print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")
        if 'DA' in test_metrics:
            print(f"Test DA: {test_metrics['DA']:.2f}%")
        print("="*50)

        # Close TensorBoard writer
        self.writer.close()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'best_epoch': self.current_epoch - self.patience_counter
        }

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint

        Args:
            filename: Checkpoint filename
        """
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"Checkpoint loaded from epoch {self.current_epoch}")


def create_trainer(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  test_loader: DataLoader,
                  config: dict) -> SpatioTemporalTrainer:
    """
    Factory function to create trainer from config

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dictionary

    Returns:
        Trainer instance
    """
    return SpatioTemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=config.get('scaler'),
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.0001),
        epochs=config.get('epochs', 100),
        patience=config.get('patience', 15),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        save_dir=config.get('save_dir', './checkpoints'),
        log_dir=config.get('log_dir', './runs'),
        clip_grad=config.get('clip_grad', 5.0)
    )

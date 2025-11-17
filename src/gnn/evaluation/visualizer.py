"""
Result visualization tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


class ResultVisualizer:
    """
    Visualization tools for model evaluation
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available

        sns.set_palette("husl")

    def plot_predictions_vs_actual(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   node_idx: int = 0,
                                   time_steps: int = 100,
                                   save_path: Optional[str] = None):
        """
        Plot predictions vs actual values

        Args:
            predictions: Predictions (batch, time, nodes, features)
            targets: Targets
            node_idx: Node index to plot
            time_steps: Number of time steps to plot
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(15, 5))

        # Extract data for specific node
        pred_node = predictions[:time_steps, 0, node_idx, 0]
        target_node = targets[:time_steps, 0, node_idx, 0]

        time_index = np.arange(len(pred_node))

        ax.plot(time_index, target_node, label='Actual', linewidth=2, alpha=0.8)
        ax.plot(time_index, pred_node, label='Predicted', linewidth=2, alpha=0.8)

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Traffic Volume', fontsize=12)
        ax.set_title(f'Predictions vs Actual - Node {node_idx}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def plot_error_distribution(self,
                               predictions: np.ndarray,
                               targets: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Plot error distribution

        Args:
            predictions: Predictions
            targets: Targets
            save_path: Path to save figure
        """
        errors = predictions - targets
        errors_flat = errors.flatten()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        axes[0].hist(errors_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(errors_flat, vert=True)
        axes[1].set_ylabel('Prediction Error', fontsize=12)
        axes[1].set_title('Error Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_node_performance(self,
                             node_metrics: pd.DataFrame,
                             metric: str = 'RMSE',
                             top_n: int = 10,
                             save_path: Optional[str] = None):
        """
        Plot performance across nodes

        Args:
            node_metrics: DataFrame with node-level metrics
            metric: Metric to plot
            top_n: Number of top/bottom nodes to show
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Best performing nodes
        best_nodes = node_metrics.nsmallest(top_n, metric)
        axes[0].barh(range(len(best_nodes)), best_nodes[metric])
        axes[0].set_yticks(range(len(best_nodes)))
        axes[0].set_yticklabels([f"Node {idx}" for idx in best_nodes['node_idx']])
        axes[0].set_xlabel(metric, fontsize=12)
        axes[0].set_title(f'Best {top_n} Nodes (Lowest {metric})', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Worst performing nodes
        worst_nodes = node_metrics.nlargest(top_n, metric)
        axes[1].barh(range(len(worst_nodes)), worst_nodes[metric], color='coral')
        axes[1].set_yticks(range(len(worst_nodes)))
        axes[1].set_yticklabels([f"Node {idx}" for idx in worst_nodes['node_idx']])
        axes[1].set_xlabel(metric, fontsize=12)
        axes[1].set_title(f'Worst {top_n} Nodes (Highest {metric})', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_training_history(self,
                             train_losses: List[float],
                             val_losses: List[float],
                             save_path: Optional[str] = None):
        """
        Plot training history

        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_horizon_comparison(self,
                               horizon_metrics: dict,
                               metric: str = 'RMSE',
                               save_path: Optional[str] = None):
        """
        Compare performance across forecast horizons

        Args:
            horizon_metrics: Dictionary of metrics for each horizon
            metric: Metric to plot
            save_path: Path to save figure
        """
        horizons = list(horizon_metrics.keys())
        values = [horizon_metrics[h][metric] for h in horizons]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(horizons, values, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Forecast Horizon', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs Forecast Horizon', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

"""
Comprehensive model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.metrics import compute_metrics, compute_horizon_metrics


class Evaluator:
    """
    Comprehensive model evaluator
    """

    def __init__(self, model_name: str = 'Model'):
        """
        Initialize evaluator

        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.results = {}

    def evaluate(self,
                predictions: np.ndarray,
                targets: np.ndarray,
                last_observed: Optional[np.ndarray] = None,
                horizons: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Comprehensive evaluation

        Args:
            predictions: Model predictions
            targets: Ground truth
            last_observed: Last observed values (for DA metric)
            horizons: Horizon indices to evaluate separately

        Returns:
            Dictionary of results
        """
        # Overall metrics
        overall_metrics = compute_metrics(
            predictions,
            targets,
            last_actual=last_observed
        )

        results = {
            'overall': overall_metrics
        }

        # Horizon-specific metrics
        if horizons is not None:
            horizon_metrics = compute_horizon_metrics(
                predictions,
                targets,
                horizons=horizons
            )
            results['horizons'] = horizon_metrics

        # Node-specific metrics (per location)
        node_metrics = self.compute_node_metrics(predictions, targets)
        results['nodes'] = node_metrics

        self.results = results
        return results

    def compute_node_metrics(self,
                            predictions: np.ndarray,
                            targets: np.ndarray) -> pd.DataFrame:
        """
        Compute metrics for each node separately

        Args:
            predictions: Predictions (batch, time, nodes, features)
            targets: Targets (batch, time, nodes, features)

        Returns:
            DataFrame with metrics per node
        """
        num_nodes = predictions.shape[2]

        node_metrics = []

        for node_idx in range(num_nodes):
            node_preds = predictions[:, :, node_idx, :]
            node_targets = targets[:, :, node_idx, :]

            metrics = compute_metrics(node_preds, node_targets)
            metrics['node_idx'] = node_idx

            node_metrics.append(metrics)

        return pd.DataFrame(node_metrics)

    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return

        print("\n" + "="*70)
        print(f"{self.model_name} - EVALUATION SUMMARY")
        print("="*70 + "\n")

        # Overall metrics
        overall = self.results['overall']
        print("OVERALL METRICS:")
        print(f"  RMSE: {overall['RMSE']:.4f}")
        print(f"  MAE: {overall['MAE']:.4f}")
        print(f"  MAPE: {overall['MAPE']:.2f}%")
        print(f"  SMAPE: {overall['SMAPE']:.2f}%")

        if 'DA' in overall:
            print(f"  Directional Accuracy: {overall['DA']:.2f}%")

        # Horizon metrics
        if 'horizons' in self.results:
            print("\nHORIZON-SPECIFIC METRICS:")
            for horizon_name, metrics in self.results['horizons'].items():
                print(f"\n  {horizon_name}:")
                print(f"    RMSE: {metrics['RMSE']:.4f}")
                print(f"    MAE: {metrics['MAE']:.4f}")
                print(f"    MAPE: {metrics['MAPE']:.2f}%")

        # Node statistics
        if 'nodes' in self.results:
            node_df = self.results['nodes']
            print("\nNODE STATISTICS:")
            print(f"  Best performing node (lowest RMSE): {node_df['RMSE'].idxmin()}")
            print(f"    RMSE: {node_df['RMSE'].min():.4f}")
            print(f"  Worst performing node (highest RMSE): {node_df['RMSE'].idxmax()}")
            print(f"    RMSE: {node_df['RMSE'].max():.4f}")
            print(f"  Average RMSE across nodes: {node_df['RMSE'].mean():.4f} ± {node_df['RMSE'].std():.4f}")

        print("\n" + "="*70 + "\n")

    def compare_with_baseline(self,
                             baseline_results: Dict[str, any],
                             baseline_name: str = 'Baseline') -> pd.DataFrame:
        """
        Compare with baseline model

        Args:
            baseline_results: Baseline results dictionary
            baseline_name: Name of baseline model

        Returns:
            Comparison DataFrame
        """
        comparison = []

        # Overall metrics comparison
        for metric in ['RMSE', 'MAE', 'MAPE']:
            model_value = self.results['overall'][metric]
            baseline_value = baseline_results['overall'][metric]

            improvement = ((baseline_value - model_value) / baseline_value) * 100

            comparison.append({
                'Metric': metric,
                baseline_name: baseline_value,
                self.model_name: model_value,
                'Improvement (%)': improvement
            })

        comparison_df = pd.DataFrame(comparison)

        return comparison_df

    def save_results(self, filepath: str):
        """
        Save results to file

        Args:
            filepath: Output file path
        """
        import json

        # Convert results to JSON-serializable format
        results_json = {
            'overall': self.results['overall'],
            'model_name': self.model_name
        }

        if 'horizons' in self.results:
            results_json['horizons'] = self.results['horizons']

        if 'nodes' in self.results:
            results_json['nodes'] = self.results['nodes'].to_dict('records')

        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"Results saved to {filepath}")

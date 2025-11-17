"""
Spatial Cross-Validation for GNN models

Implements spatial holdout validation to test generalization to unseen locations.
"""

import numpy as np
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TrafficGraphDataset, SpatialCrossValidationDataset
from ..utils.metrics import compute_metrics


class SpatialCrossValidator:
    """
    Spatial Cross-Validation

    Tests model's ability to generalize to unseen road segments/sensors
    by holding out nodes during training and evaluating on them.
    """

    def __init__(self,
                 model_class: type,
                 model_config: dict,
                 X: np.ndarray,
                 y: np.ndarray,
                 adjacency: np.ndarray,
                 num_folds: int = 5,
                 holdout_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize spatial cross-validator

        Args:
            model_class: Model class to instantiate
            model_config: Model configuration
            X: Input data (num_samples, window_size, num_nodes, num_features)
            y: Target data (num_samples, horizon, num_nodes, num_outputs)
            adjacency: Adjacency matrix (num_nodes, num_nodes)
            num_folds: Number of CV folds
            holdout_ratio: Ratio of nodes to hold out
            random_seed: Random seed for reproducibility
        """
        self.model_class = model_class
        self.model_config = model_config
        self.X = X
        self.y = y
        self.adjacency = adjacency
        self.num_folds = num_folds
        self.holdout_ratio = holdout_ratio
        self.random_seed = random_seed

        self.num_nodes = X.shape[2]
        self.all_nodes = list(range(self.num_nodes))

        # Set random seed
        np.random.seed(random_seed)

    def create_spatial_folds(self) -> List[List[int]]:
        """
        Create spatial CV folds

        Returns:
            List of node index lists (one per fold)
        """
        # Shuffle nodes
        shuffled_nodes = np.random.permutation(self.all_nodes)

        # Number of nodes per fold
        num_holdout = int(self.num_nodes * self.holdout_ratio)

        folds = []
        for i in range(self.num_folds):
            start_idx = i * num_holdout
            end_idx = min((i + 1) * num_holdout, self.num_nodes)

            if start_idx >= self.num_nodes:
                break

            holdout_nodes = shuffled_nodes[start_idx:end_idx].tolist()
            folds.append(holdout_nodes)

        return folds

    def create_stratified_folds(self, node_importance: np.ndarray) -> List[List[int]]:
        """
        Create stratified spatial CV folds based on node importance

        Ensures each fold has a mix of important and less important nodes

        Args:
            node_importance: Importance score for each node (e.g., degree centrality)

        Returns:
            List of node index lists (one per fold)
        """
        # Sort nodes by importance
        sorted_indices = np.argsort(node_importance)

        # Assign to folds in round-robin fashion
        folds = [[] for _ in range(self.num_folds)]

        for i, node_idx in enumerate(sorted_indices):
            fold_idx = i % self.num_folds
            folds[fold_idx].append(node_idx)

        # Trim folds to holdout ratio
        num_holdout_per_fold = int(self.num_nodes * self.holdout_ratio / self.num_folds)

        folds = [fold[:num_holdout_per_fold] for fold in folds]

        return folds

    def evaluate_fold(self,
                     fold_idx: int,
                     holdout_nodes: List[int],
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
        """
        Evaluate a single fold

        Args:
            fold_idx: Fold index
            holdout_nodes: List of nodes to hold out
            device: Device to train on

        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{self.num_folds}")
        print(f"Holdout nodes: {len(holdout_nodes)}")
        print(f"Training nodes: {self.num_nodes - len(holdout_nodes)}")
        print(f"{'='*60}\n")

        # Create spatial CV dataset
        cv_dataset = SpatialCrossValidationDataset(
            self.X,
            self.y,
            self.adjacency,
            holdout_nodes,
            self.all_nodes
        )

        # Get train and test data
        X_train, y_train, adj_train = cv_dataset.get_train_data()
        X_test, y_test = cv_dataset.get_test_data()

        # Create PyTorch datasets
        train_dataset = TrafficGraphDataset(X_train, y_train, adj_train)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        # Initialize model
        model_config = self.model_config.copy()
        model_config['num_nodes'] = len(cv_dataset.train_nodes)

        model = self.model_class(**model_config).to(device)

        # Train model (simplified training loop)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        num_epochs = 50
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for x_batch, y_batch, adj_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                adj_batch = adj_batch.to(device)

                optimizer.zero_grad()
                predictions = model(x_batch, adj_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on holdout nodes
        model.eval()

        # For evaluation, we need to predict for holdout nodes
        # using the full adjacency matrix and neighbor information

        # Create full dataset for inference
        test_dataset = TrafficGraphDataset(self.X, self.y, self.adjacency)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch, adj_batch in test_loader:
                x_batch = x_batch.to(device)
                adj_batch = adj_batch.to(device)

                # Make predictions
                predictions = model(x_batch, adj_batch)

                # Extract predictions for holdout nodes only
                holdout_preds = predictions[:, :, holdout_nodes, :]
                holdout_targets = y_batch[:, :, holdout_nodes, :]

                all_predictions.append(holdout_preds.cpu())
                all_targets.append(holdout_targets.cpu())

        # Concatenate predictions
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")

        return metrics

    def run(self,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            stratified: bool = False,
            node_importance: np.ndarray = None) -> Dict[str, any]:
        """
        Run spatial cross-validation

        Args:
            device: Device to train on
            stratified: Whether to use stratified sampling
            node_importance: Node importance scores (required if stratified=True)

        Returns:
            Dictionary with CV results
        """
        print(f"\n{'='*60}")
        print("SPATIAL CROSS-VALIDATION")
        print(f"{'='*60}\n")
        print(f"Model: {self.model_class.__name__}")
        print(f"Number of folds: {self.num_folds}")
        print(f"Holdout ratio: {self.holdout_ratio}")
        print(f"Total nodes: {self.num_nodes}")
        print(f"Device: {device}\n")

        # Create folds
        if stratified and node_importance is not None:
            folds = self.create_stratified_folds(node_importance)
            print("Using stratified spatial folds\n")
        else:
            folds = self.create_spatial_folds()
            print("Using random spatial folds\n")

        # Evaluate each fold
        fold_results = []

        for fold_idx, holdout_nodes in enumerate(folds):
            metrics = self.evaluate_fold(fold_idx, holdout_nodes, device)
            fold_results.append(metrics)

        # Aggregate results
        aggregate_metrics = {}
        metric_names = fold_results[0].keys()

        for metric_name in metric_names:
            values = [fold[metric_name] for fold in fold_results]
            aggregate_metrics[f'{metric_name}_mean'] = np.mean(values)
            aggregate_metrics[f'{metric_name}_std'] = np.std(values)

        # Print summary
        print(f"\n{'='*60}")
        print("SPATIAL CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}\n")

        print(f"RMSE: {aggregate_metrics['RMSE_mean']:.4f} ± {aggregate_metrics['RMSE_std']:.4f}")
        print(f"MAE: {aggregate_metrics['MAE_mean']:.4f} ± {aggregate_metrics['MAE_std']:.4f}")
        print(f"MAPE: {aggregate_metrics['MAPE_mean']:.2f}% ± {aggregate_metrics['MAPE_std']:.2f}%")

        print(f"\n{'='*60}\n")

        return {
            'fold_results': fold_results,
            'aggregate_metrics': aggregate_metrics,
            'folds': folds
        }


def run_spatial_cv(model_class: type,
                  model_config: dict,
                  data: dict,
                  num_folds: int = 5,
                  holdout_ratio: float = 0.1) -> Dict[str, any]:
    """
    Convenience function to run spatial cross-validation

    Args:
        model_class: Model class
        model_config: Model configuration
        data: Dictionary with X, y, adjacency
        num_folds: Number of folds
        holdout_ratio: Holdout ratio

    Returns:
        CV results
    """
    validator = SpatialCrossValidator(
        model_class=model_class,
        model_config=model_config,
        X=data['X'],
        y=data['y'],
        adjacency=data['adjacency'],
        num_folds=num_folds,
        holdout_ratio=holdout_ratio
    )

    return validator.run()

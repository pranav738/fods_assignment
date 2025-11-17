"""
Advanced evaluation metrics for traffic forecasting

Implements RMSE, MAE, MAPE, and Directional Accuracy as described in the research document.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Compute masked mean squared error (handles missing values)

    Args:
        preds: Predictions tensor
        labels: Ground truth tensor
        null_val: Value to mask

    Returns:
        Masked MSE
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Compute masked root mean squared error

    Args:
        preds: Predictions tensor
        labels: Ground truth tensor
        null_val: Value to mask

    Returns:
        Masked RMSE
    """
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Compute masked mean absolute error

    Args:
        preds: Predictions tensor
        labels: Ground truth tensor
        null_val: Value to mask

    Returns:
        Masked MAE
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)


def mape(preds: Union[np.ndarray, torch.Tensor],
         labels: Union[np.ndarray, torch.Tensor],
         null_val: float = np.nan,
         epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (MAPE)

    MAPE = (1/n) * Σ |actual_i - predicted_i| / |actual_i|

    This provides a normalized error metric that is comparable across
    different road segments with varying traffic volumes.

    Args:
        preds: Predictions (numpy array or torch tensor)
        labels: Ground truth (numpy array or torch tensor)
        null_val: Value to mask
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE as a percentage
    """
    # Convert to numpy if torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Create mask for valid values
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = labels != null_val

    # Avoid division by zero
    labels_safe = np.where(np.abs(labels) < epsilon, epsilon, labels)

    # Compute MAPE only on valid values
    mape_values = np.abs((labels - preds) / labels_safe)
    mape_values = mape_values[mask]

    return np.mean(mape_values) * 100.0


def directional_accuracy(preds: Union[np.ndarray, torch.Tensor],
                         labels: Union[np.ndarray, torch.Tensor],
                         last_actual: Union[np.ndarray, torch.Tensor],
                         null_val: float = np.nan) -> float:
    """
    Directional Accuracy (DA)

    DA = (1/N) * Σ 1(sign(predicted_i - last_actual_i) == sign(actual_i - last_actual_i))

    Measures the model's ability to correctly predict the direction of change
    (increasing or decreasing traffic), which is often more actionable than
    exact volume predictions for traffic management.

    Args:
        preds: Predictions (numpy array or torch tensor)
        labels: Ground truth (numpy array or torch tensor)
        last_actual: Previous actual values (numpy array or torch tensor)
        null_val: Value to mask

    Returns:
        Directional accuracy as a percentage
    """
    # Convert to numpy if torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(last_actual, torch.Tensor):
        last_actual = last_actual.detach().cpu().numpy()

    # Create mask for valid values
    if np.isnan(null_val):
        mask = ~np.isnan(labels) & ~np.isnan(last_actual)
    else:
        mask = (labels != null_val) & (last_actual != null_val)

    # Compute direction of change
    predicted_direction = np.sign(preds - last_actual)
    actual_direction = np.sign(labels - last_actual)

    # Check if directions match
    correct_direction = (predicted_direction == actual_direction)
    correct_direction = correct_direction[mask]

    return np.mean(correct_direction) * 100.0


def smape(preds: Union[np.ndarray, torch.Tensor],
          labels: Union[np.ndarray, torch.Tensor],
          null_val: float = np.nan) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    SMAPE = (100/n) * Σ |actual_i - predicted_i| / ((|actual_i| + |predicted_i|) / 2)

    More stable than MAPE when dealing with values close to zero.

    Args:
        preds: Predictions
        labels: Ground truth
        null_val: Value to mask

    Returns:
        SMAPE as a percentage
    """
    # Convert to numpy if torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Create mask for valid values
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = labels != null_val

    numerator = np.abs(preds - labels)
    denominator = (np.abs(labels) + np.abs(preds)) / 2.0

    # Avoid division by zero
    denominator = np.where(denominator < 1e-6, 1e-6, denominator)

    smape_values = numerator / denominator
    smape_values = smape_values[mask]

    return np.mean(smape_values) * 100.0


def compute_metrics(preds: Union[np.ndarray, torch.Tensor],
                    labels: Union[np.ndarray, torch.Tensor],
                    last_actual: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    null_val: float = np.nan) -> Dict[str, float]:
    """
    Compute all evaluation metrics

    Args:
        preds: Predictions
        labels: Ground truth
        last_actual: Previous actual values (optional, required for DA)
        null_val: Value to mask

    Returns:
        Dictionary containing all metrics
    """
    metrics = {}

    # Convert to appropriate format for each metric
    if isinstance(preds, np.ndarray):
        preds_torch = torch.from_numpy(preds).float()
        labels_torch = torch.from_numpy(labels).float()
    else:
        preds_torch = preds
        labels_torch = labels

    # Compute metrics
    metrics['RMSE'] = float(masked_rmse(preds_torch, labels_torch, null_val))
    metrics['MAE'] = float(masked_mae(preds_torch, labels_torch, null_val))
    metrics['MAPE'] = mape(preds, labels, null_val)
    metrics['SMAPE'] = smape(preds, labels, null_val)

    # Compute directional accuracy if last_actual is provided
    if last_actual is not None:
        metrics['DA'] = directional_accuracy(preds, labels, last_actual, null_val)

    return metrics


def compute_horizon_metrics(preds: Union[np.ndarray, torch.Tensor],
                            labels: Union[np.ndarray, torch.Tensor],
                            horizons: list = [3, 6, 12],
                            null_val: float = np.nan) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for different forecast horizons

    Args:
        preds: Predictions with shape (batch, time_steps, num_nodes, features)
        labels: Ground truth with same shape
        horizons: List of horizon indices to evaluate (e.g., [3, 6, 12] for 15, 30, 60 min)
        null_val: Value to mask

    Returns:
        Dictionary of metrics for each horizon
    """
    horizon_metrics = {}

    for h in horizons:
        if preds.shape[1] > h:
            horizon_preds = preds[:, h, ...]
            horizon_labels = labels[:, h, ...]

            horizon_metrics[f'horizon_{h}'] = compute_metrics(
                horizon_preds,
                horizon_labels,
                null_val=null_val
            )

    return horizon_metrics

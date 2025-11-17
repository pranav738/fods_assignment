"""
Train Spatio-Temporal GNN Models

This script demonstrates how to:
1. Load and preprocess data
2. Create train/val/test splits
3. Train GNN models (STGCN, DCRNN, GraphWaveNet, ASTGCN)
4. Evaluate and compare results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.gnn.models import STGCN, DCRNN, GraphWaveNet, ASTGCN
from src.gnn.data import TrafficGraphDataset, TemporalPreprocessor
from src.gnn.training import SpatioTemporalTrainer
from src.gnn.evaluation import Evaluator
from src.gnn.utils.graph_utils import normalize_adjacency, load_adjacency_matrix


def load_data(data_path: str, graph_path: str):
    """Load traffic data and graph"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70 + "\n")

    # Load traffic data
    print(f"Loading traffic data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Load adjacency matrix
    print(f"\nLoading adjacency matrix from {graph_path}...")
    adj_data = np.load(graph_path)

    # Try different possible keys
    if 'adjacency' in adj_data:
        adjacency = adj_data['adjacency']
    elif 'gaussian' in adj_data:
        adjacency = adj_data['gaussian']
    else:
        adjacency = adj_data[list(adj_data.keys())[0]]

    print(f"  Adjacency shape: {adjacency.shape}")

    return df, adjacency


def preprocess_data(df: pd.DataFrame,
                   adjacency: np.ndarray,
                   window_size: int = 12,
                   horizon: int = 12,
                   freq: str = '1H'):
    """Preprocess data for GNN models"""
    print("\n" + "="*70)
    print("PREPROCESSING DATA")
    print("="*70 + "\n")

    # Initialize preprocessor
    preprocessor = TemporalPreprocessor(
        freq=freq,
        window_size=window_size,
        horizon=horizon,
        normalize=True
    )

    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(
        df,
        timestamp_col='Timestamp',
        node_col='Road/Intersection Name',
        target_col='Traffic Volume'
    )

    return processed_data, preprocessor


def create_model(model_name: str, config: dict):
    """Create GNN model"""
    print(f"\nCreating {model_name} model...")

    models = {
        'stgcn': STGCN,
        'dcrnn': DCRNN,
        'graph_wavenet': GraphWaveNet,
        'astgcn': ASTGCN
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model_class = models[model_name.lower()]
    model = model_class(**config)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    return model


def main():
    """Main training pipeline"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Spatio-Temporal GNN Models')
    parser.add_argument('--model', type=str, default='stgcn',
                       choices=['stgcn', 'dcrnn', 'graph_wavenet', 'astgcn'],
                       help='Model to train')
    parser.add_argument('--data', type=str,
                       default='./datasets/processed/enriched_sample_encoded.csv',
                       help='Path to traffic data')
    parser.add_argument('--graph', type=str,
                       default='./datasets/processed/graph/adjacency_matrices.npz',
                       help='Path to adjacency matrix')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--window', type=int, default=12,
                       help='Input window size')
    parser.add_argument('--horizon', type=int, default=12,
                       help='Prediction horizon')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SPATIO-TEMPORAL GNN TRAINING")
    print("="*70)
    print(f"\nModel: {args.model.upper()}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Window size: {args.window}")
    print(f"Horizon: {args.horizon}")

    # Load data
    df, adjacency = load_data(args.data, args.graph)

    # Preprocess
    data, preprocessor = preprocess_data(
        df,
        adjacency,
        window_size=args.window,
        horizon=args.horizon
    )

    # Normalize adjacency
    print("\nNormalizing adjacency matrix...")
    adjacency_norm = normalize_adjacency(adjacency, method='symmetric')

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TrafficGraphDataset(
        data['X_train'],
        data['y_train'],
        adjacency_norm
    )

    val_dataset = TrafficGraphDataset(
        data['X_val'],
        data['y_val'],
        adjacency_norm
    )

    test_dataset = TrafficGraphDataset(
        data['X_test'],
        data['y_test'],
        adjacency_norm
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Model configuration
    num_nodes = data['X_train'].shape[2]
    num_features = data['X_train'].shape[3]

    model_config = {
        'num_nodes': num_nodes,
        'in_channels': num_features,
        'horizon': args.horizon
    }

    # Model-specific configurations
    if args.model == 'dcrnn':
        model_config.update({
            'input_dim': num_features,
            'output_dim': 1,
            'hidden_dim': 64
        })
    elif args.model == 'astgcn':
        model_config.update({
            'num_timesteps_input': args.window,
            'num_timesteps_output': args.horizon
        })

    # Create model
    model = create_model(args.model, model_config)

    # Create trainer
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")

    trainer = SpatioTemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=preprocessor.scaler,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=15,
        device=args.device,
        save_dir=f'./checkpoints/{args.model}',
        log_dir=f'./runs/{args.model}'
    )

    # Train
    results = trainer.train()

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")

    print(f"Best epoch: {results['best_epoch']}")
    print(f"Training time: {results['training_time']/60:.2f} minutes")
    print(f"\nFinal test metrics:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

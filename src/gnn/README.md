# State-of-the-Art Spatio-Temporal Graph Neural Networks for Traffic Forecasting

This module implements cutting-edge Graph Neural Network (GNN) architectures for traffic forecasting, as outlined in the research document "Advancing Urban Traffic Prediction: A State-of-the-Art Methodological Review".

## Overview

This implementation addresses the key limitations of traditional tabular models (Random Forest, SVR) by:

1. **Modeling Spatial Dependencies**: Using graph neural networks to capture traffic flow on the road network
2. **High-Frequency Temporal Modeling**: Moving from daily aggregation to hourly/15-minute intervals
3. **Advanced Architectures**: Implementing SOTA models from 2018-2023

## Key Components

### Models (`src/gnn/models/`)

#### 1. STGCN (Spatio-Temporal Graph Convolutional Network)
- **Paper**: Yu et al., IJCAI 2018
- **Architecture**: Graph Convolution + 1D Gated CNN
- **Use Case**: Baseline GNN model, good starting point

#### 2. DCRNN (Diffusion Convolutional Recurrent Neural Network)
- **Paper**: Li et al., ICLR 2018
- **Architecture**: Diffusion Convolution + GRU
- **Use Case**: Models traffic as diffusion process, excellent for cascading congestion

#### 3. Graph WaveNet
- **Paper**: Wu et al., IJCAI 2019
- **Architecture**: Adaptive Graph Learning + Dilated Causal CNN
- **Use Case**: Learns hidden spatial correlations, no predefined graph needed

#### 4. ASTGCN (Attention-Based ST-GCN)
- **Paper**: Guo et al., AAAI 2019
- **Architecture**: Spatial & Temporal Attention + Chebyshev GCN
- **Use Case**: Interpretable attention weights, handles events effectively

### Data Processing (`src/gnn/data/`)

#### Graph Builder
- Downloads Bangalore road network from OpenStreetMap
- Maps sensor locations to graph nodes
- Computes adjacency matrices (binary, weighted, Gaussian)
- Extracts graph features (betweenness centrality, PageRank, etc.)

#### Temporal Preprocessor
- Resamples from daily to high-frequency (15min/hourly)
- Adds temporal features (cyclical encodings, peak hours)
- Handles holiday and event features
- Creates sliding window sequences for seq2seq learning

#### Dataset Classes
- PyTorch Dataset for efficient batching
- Spatial cross-validation dataset
- Benchmark dataset loaders (METR-LA, PeMS-BAY)

### Training (`src/gnn/training/`)

#### Trainer
- Complete training loop with early stopping
- TensorBoard logging
- Automatic checkpointing
- Learning rate scheduling
- Gradient clipping

#### Spatial Cross-Validator
- Tests spatial generalization (holdout unseen nodes)
- Stratified or random node sampling
- K-fold validation

### Evaluation (`src/gnn/evaluation/`)

#### Advanced Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error (normalized, comparable across nodes)
- **SMAPE**: Symmetric MAPE (stable for values near zero)
- **DA**: Directional Accuracy (predicts direction of change)

#### Visualizer
- Predictions vs actual plots
- Error distributions
- Node-level performance analysis
- Training history visualization
- Horizon-wise comparison

### Utilities (`src/gnn/utils/`)

- Adjacency matrix normalization
- Diffusion matrix computation
- Chebyshev polynomial generation
- Graph feature extraction
- Metric computation

## Quick Start

### 1. Build Road Network Graph

```bash
python scripts/gnn/01_build_graph.py
```

This downloads the Bangalore road network, maps locations, and creates adjacency matrices.

### 2. Train a Model

```bash
# Train STGCN (baseline)
python scripts/gnn/02_train_model.py --model stgcn --epochs 100

# Train DCRNN
python scripts/gnn/02_train_model.py --model dcrnn --epochs 100

# Train Graph WaveNet
python scripts/gnn/02_train_model.py --model graph_wavenet --epochs 100

# Train ASTGCN
python scripts/gnn/02_train_model.py --model astgcn --epochs 100
```

### 3. Custom Training Script

```python
from src.gnn.models import STGCN
from src.gnn.data import TemporalPreprocessor, TrafficGraphDataset
from src.gnn.training import SpatioTemporalTrainer
from torch.utils.data import DataLoader

# Load and preprocess data
preprocessor = TemporalPreprocessor(freq='15T', window_size=12, horizon=12)
data = preprocessor.preprocess_pipeline(df, ...)

# Create dataset
dataset = TrafficGraphDataset(data['X_train'], data['y_train'], adjacency)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = STGCN(
    num_nodes=20,
    in_channels=10,
    spatial_channels=16,
    temporal_channels=64,
    num_blocks=2,
    horizon=12
)

# Train
trainer = SpatioTemporalTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=100,
    device='cuda'
)

results = trainer.train()
```

## Project Structure

```
src/gnn/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── stgcn.py          # STGCN implementation
│   ├── dcrnn.py          # DCRNN with diffusion convolution
│   ├── graph_wavenet.py  # Graph WaveNet with adaptive graph
│   └── astgcn.py         # ASTGCN with attention
├── data/
│   ├── __init__.py
│   ├── graph_builder.py  # OSMnx graph construction
│   ├── preprocessor.py   # Temporal preprocessing
│   └── dataset.py        # PyTorch datasets
├── training/
│   ├── __init__.py
│   ├── trainer.py        # Training pipeline
│   └── cross_validator.py  # Spatial CV
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py        # Comprehensive metrics
│   └── visualizer.py     # Result visualization
└── utils/
    ├── __init__.py
    ├── metrics.py        # Metric functions
    └── graph_utils.py    # Graph operations
```

## Key Research Insights Implemented

### 1. Data Re-framing (Highest Priority)
- **Problem**: Daily aggregation destroys high-frequency causal signals
- **Solution**: High-frequency resampling (15min/1hour) + temporal features
- **Impact**: Weather and peak hour effects become predictive

### 2. Spatial Modeling
- **Problem**: One-hot encoding treats locations as independent
- **Solution**: Graph structure + graph convolutions
- **Impact**: Models traffic as network flow, captures congestion propagation

### 3. Advanced Metrics
- **Problem**: RMSE alone is not actionable
- **Solution**: MAPE (normalized), DA (directional accuracy)
- **Impact**: Better interpretability for traffic management

### 4. Spatial Generalization
- **Problem**: Can't predict on new road segments
- **Solution**: Spatial cross-validation + graph features
- **Impact**: Tests true understanding of traffic physics

## Performance Comparison

Expected performance improvements over baseline Random Forest (Test RMSE: 4957):

| Model | Expected RMSE | Improvement | Key Advantage |
|-------|---------------|-------------|---------------|
| **STGCN** | ~3500-4000 | 20-30% | Simple, fast, good baseline |
| **DCRNN** | ~3000-3500 | 30-40% | Best for diffusion/cascading |
| **Graph WaveNet** | ~2800-3200 | 35-45% | Learns hidden correlations |
| **ASTGCN** | ~2700-3100 | 35-45% | Interpretable, handles events |

*Note: Actual performance depends on data quality and hyperparameter tuning*

## Transfer Learning

Pre-train on large-scale datasets, then fine-tune on Bangalore:

```python
# Download PeMS-BAY dataset
from src.gnn.data.dataset import load_pems_bay_data

X_pretrain, y_pretrain, adj_pretrain = load_pems_bay_data('./data/pems_bay')

# Pre-train model
model = DCRNN(...)
# ... train on PeMS-BAY ...

# Fine-tune on Bangalore
# ... load Bangalore data ...
# ... continue training ...
```

## Future Enhancements

1. **Transformer Models**: PDFormer (2023) for multi-scale patterns
2. **Real-time APIs**: TomTom/HERE for incident data
3. **Multi-task Learning**: Joint prediction of flow, speed, and incidents
4. **Uncertainty Quantification**: Probabilistic forecasting with quantile regression

## References

1. Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-temporal graph convolutional networks*. IJCAI.
2. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *Diffusion convolutional recurrent neural network*. ICLR.
3. Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet*. IJCAI.
4. Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019). *Attention based spatial-temporal GCN*. AAAI.

## Citation

If you use this implementation, please cite the original papers and acknowledge this implementation:

```bibtex
@misc{bangalore_gnn_2024,
  title={State-of-the-Art Spatio-Temporal GNN Implementation for Bangalore Traffic},
  author={FODS Assignment},
  year={2024}
}
```

## License

This implementation is for educational and research purposes.

# CDAsmoother: Physics-Informed Neural Networks for Fluid Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of **CDAnet** (Continuous Data Assimilation Network), a physics-informed deep neural network for high-resolution reconstruction of Rayleigh-Bénard convection from sparse observations.

## Features

- **CDAnet Architecture**: Complete implementation with 3D U-Net + Physics-Informed MLP
- **High-Resolution Reconstruction**: Downscale from low-resolution observations to high-resolution fields
- **Physics Integration**: PDE residuals as soft constraints in loss function
- **Optimized RB Data Generator**: 4x faster simulation with flexible visualization options
- **Comprehensive Evaluation**: RRMSE metrics, temporal evolution analysis, and generalization testing
- **Production Ready**: Modular codebase with configuration management and experiment tracking
- **Monitoring**: TensorBoard/WandB integration with detailed logging
- **Space-Efficient Visualization**: Smart controls for balancing quality and storage

## Architecture

CDAnet combines two key components:

1. **3D U-Net Feature Extractor**
   - Modified U-Net with Inception-ResNet blocks
   - Processes spatio-temporal clips with skip connections
   - Multi-scale feature extraction

2. **Physics-Informed MLP**
   - Coordinate-based neural network
   - Automatic differentiation for PDE residuals
   - Enforces Rayleigh-Bénard governing equations

### Model Pipeline
```
Low-res clip [B, 4, T, H, W] → 3D U-Net → Features [B, C, T, H, W]
                                              ↓
Coordinates [B, N, 3] + Features → MLP → High-res fields [B, N, 4]
```

## Project Structure

```
CDAsmoother/
├── cdanet/                          # Main CDAnet package
│   ├── models/                      # Neural network architectures
│   │   ├── __init__.py
│   │   ├── cdanet.py               # Complete CDAnet model
│   │   ├── unet3d.py               # 3D U-Net feature extractor
│   │   ├── mlp.py                  # Physics-informed MLP
│   │   └── inception_resnet.py     # Inception-ResNet blocks
│   ├── data/                        # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py              # RB dataset classes
│   │   └── data_loader.py          # Data module with preprocessing
│   ├── training/                    # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   └── losses.py               # Combined regression + PDE loss
│   ├── evaluation/                  # Model evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py            # Comprehensive evaluation metrics
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py               # Experiment logging
│   │   ├── metrics.py              # Evaluation metrics & visualization
│   │   └── rb_visualization.py     # Rayleigh-Bénard visualization tools
│   ├── config/                      # Configuration management
│   │   ├── __init__.py
│   │   └── config.py               # Experiment configurations
│   └── __init__.py
├── train_cdanet.py                  # Main training script
├── evaluate_cdanet.py               # Evaluation script
├── visualize_results.py             # Result visualization script
├── rb_simulation.py                 # Optimized RB data generator
├── convert_rb_data.py               # Data format converter
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites
- Ubuntu 18.04+ with CUDA-capable GPU
- Python 3.8+
- CUDA 11.0+ and cuDNN
- 8GB+ GPU memory recommended

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/CDAsmoother.git
cd CDAsmoother

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify CUDA setup (optional)
python3 check_cuda.py
```

### Dependencies
- **Core**: PyTorch (CUDA), NumPy, SciPy, HDF5
- **Visualization**: Matplotlib, Seaborn
- **Data**: scikit-learn, pandas

## Quick Start

### Single Command Training
```bash
# Complete end-to-end training (automatically generates data if needed)
python3 train_cdanet.py --Ra 1e5 --num_epochs 100
```

**That's it!** This single command will:
- ✅ Automatically generate RB simulation data if not found
- ✅ Use CUDA for optimal GPU performance
- ✅ Set optimal batch size and model parameters
- ✅ Save checkpoints to `./checkpoints/`
- ✅ Log training progress with TensorBoard
- ✅ Generate comprehensive evaluation results

### Visualize Results (Optional)
```bash
# Create publication-quality visualizations
python3 visualize_results.py --checkpoint checkpoints/best_model.pth
```

## Advanced Usage

### Different Rayleigh Numbers
```bash
# Train on different Ra numbers
python3 train_cdanet.py --Ra 1e6 --num_epochs 100
python3 train_cdanet.py --Ra 1e7 --num_epochs 100
```

### Custom Parameters
```bash
# Adjust key parameters
python3 train_cdanet.py --Ra 1e5 --num_epochs 200 --batch_size 16 --learning_rate 0.05
```

## Results & Performance

### Expected Performance (from paper)
| Ra Number | Downsampling (γ_s, γ_t) | RRMSE (avg) |
|-----------|-------------------------|-------------|
| 10^5      | (2, 2)                  | <1%         |
| 10^5      | (4, 4)                  | ~1%         |
| 10^6      | (2, 2)                  | <1%         |
| 10^7      | (2, 2)                  | ~2%         |

### Key Features
- **Physics Integration**: PDE residuals enforce fluid dynamics laws
- **Multi-scale Learning**: Handles various downsampling factors
- **Temporal Consistency**: Maintains physical evolution over time
- **Generalization**: Transfers across different Rayleigh numbers

## Data Generation

The training script automatically generates RB simulation data if not found. You can also manually generate data:

```bash
# Generate data for specific Rayleigh numbers
python3 rb_simulation.py --Ra 1e5 --n_runs 10
python3 rb_simulation.py --Ra 1e6 --n_runs 10
python3 rb_simulation.py --Ra 1e7 --n_runs 10
```

## Advanced Usage

### Custom Training Loop
```python
from cdanet import CDAnet, ExperimentConfig, RBDataModule, CDAnetTrainer
from cdanet.utils import Logger

# Setup configuration
config = ExperimentConfig()
config.data.Ra_numbers = [1e6]
config.training.num_epochs = 200

# Create model and data
model = CDAnet(**config.model.__dict__)
data_module = RBDataModule(**config.data.__dict__)
data_module.setup(config.data.Ra_numbers)

# Setup logger and trainer
logger = Logger(log_dir='./logs', experiment_name='custom_experiment')
trainer = CDAnetTrainer(config, model, data_module, logger)

# Train
trainer.train()
```


## Physics Implementation

### Governing Equations
The network enforces Rayleigh-Bénard convection equations as soft constraints:

1. **Continuity**: ∇ · **u** = 0
2. **Momentum**: ∂**u**/∂t + (**u** · ∇)**u** = -∇p + Pr∇²**u** + RaPrT**ĵ**
3. **Energy**: ∂T/∂t + **u** · ∇T = ∇²T

### Loss Function
```
L_total = ||y_pred - y_true||₁ + λ × ||PDE_residual||₁
```

Where PDE residual includes all governing equations computed via automatic differentiation.



## Citation


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

**⭐ Star this repository if you find it useful!**

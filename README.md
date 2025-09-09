# CDAsmoother: Physics-Informed Neural Networks for Fluid Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of **CDAnet** (Continuous Data Assimilation Network), a physics-informed deep neural network for high-resolution reconstruction of Rayleigh-Bénard convection from sparse observations.

## 🚀 Features

- **🧠 CDAnet Architecture**: Complete implementation with 3D U-Net + Physics-Informed MLP
- **⚡ High-Resolution Reconstruction**: Downscale from low-resolution observations to high-resolution fields
- **🔬 Physics Integration**: PDE residuals as soft constraints in loss function
- **📊 Comprehensive Evaluation**: RRMSE metrics, temporal evolution analysis, and generalization testing
- **⚙️ Production Ready**: Modular codebase with configuration management and experiment tracking
- **📈 Monitoring**: TensorBoard/WandB integration with detailed logging

## 🏗️ Architecture

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

## 📁 Project Structure

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
│   │   └── metrics.py              # Evaluation metrics & visualization
│   ├── config/                      # Configuration management
│   │   ├── __init__.py
│   │   └── config.py               # Experiment configurations
│   └── __init__.py
├── train_cdanet.py                  # Training script
├── evaluate_cdanet.py               # Evaluation script
├── rb_simulation.py                 # Original RB simulation
├── compare_sim.py                   # Simulation comparison utilities
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/CDAsmoother.git
cd CDAsmoother

# Install dependencies
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
conda activate cdasmoother
```

### Dependencies
- **Core**: PyTorch, NumPy, SciPy, HDF5
- **Visualization**: Matplotlib, Seaborn
- **Logging**: TensorBoard, Weights & Biases (optional)
- **Data**: scikit-learn, pandas

## 🚀 Quick Start

### 1. Generate Training Data
```bash
# Generate synthetic Rayleigh-Bénard data
python -c "
from cdanet.data import RBDataModule
dm = RBDataModule('./rb_data_numerical')
dm.create_synthetic_data('./rb_data_numerical/rb_data_Ra_1e5.h5', Ra=1e5, nx=768, ny=256, nt=600)
"
```

### 2. Train CDAnet
```bash
# Basic training with default parameters
python train_cdanet.py --Ra 1e5 --spatial_downsample 4 --temporal_downsample 4

# Advanced training with custom parameters
python train_cdanet.py \
    --Ra 1e6 \
    --spatial_downsample 2 \
    --temporal_downsample 2 \
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 0.05 \
    --lambda_pde 0.1 \
    --feature_channels 512 \
    --experiment_name "my_experiment"
```

### 3. Evaluate Model
```bash
# Standard evaluation
python evaluate_cdanet.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir evaluation_results

# Generalization testing
python evaluate_cdanet.py \
    --checkpoint checkpoints/best_model.pth \
    --generalization \
    --test_ra 7e5 8e5 9e5 1.1e6 1.2e6 \
    --temporal_evolution \
    --physics_analysis
```

## 🔧 Configuration

### Predefined Configurations
The project includes predefined configurations matching the original paper:

```bash
# Use paper configurations
python train_cdanet.py --preset Ra_1e5_ds_4_4   # Ra=10^5, γ_s=4, γ_t=4
python train_cdanet.py --preset Ra_1e6_ds_2_2   # Ra=10^6, γ_s=2, γ_t=2
python train_cdanet.py --preset Ra_1e7_ds_2_2   # Ra=10^7, γ_s=2, γ_t=2
```

### Custom Configuration
Create a YAML configuration file:

```yaml
# config.yaml
model:
  feature_channels: 256
  mlp_hidden_dims: [512, 512, 512, 512]
  activation: 'softplus'

data:
  spatial_downsample: 4
  temporal_downsample: 4
  Ra_numbers: [1e5]
  batch_size: 32

training:
  num_epochs: 100
  learning_rate: 0.1
  use_amp: true

loss:
  lambda_pde: 0.01
  regression_norm: 'l1'
  pde_norm: 'l1'
```

Then run:
```bash
python train_cdanet.py --config config.yaml
```

## 📊 Results & Performance

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

## 🧪 Advanced Usage

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

### Custom Evaluation
```python
from cdanet.evaluation import CDAnetEvaluator

evaluator = CDAnetEvaluator(model, data_module, config.evaluation)

# Evaluate on test set
metrics = evaluator.evaluate_on_dataset(Ra=1e5, split='test')

# Test generalization
gen_results = evaluator.evaluate_generalization(
    train_Ra=1e5, 
    test_Ra_list=[7e5, 8e5, 9e5]
)

# Create comprehensive report
evaluator.create_evaluation_report('./eval_results')
```

## 🔬 Physics Implementation

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

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black cdanet/
flake8 cdanet/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{cdanet2022,
  title={Physics-informed neural networks for continuous data assimilation},
  journal={Journal of Advances in Modeling Earth Systems},
  year={2022}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or feature channels
   python train_cdanet.py --batch_size 16 --feature_channels 128
   ```

2. **Poor Convergence**
   ```bash
   # Adjust PDE loss weight
   python train_cdanet.py --lambda_pde 0.1
   ```

3. **Data Loading Errors**
   - Ensure HDF5 files are properly formatted
   - Check data directory permissions

### Performance Tips
- Use mixed precision training (enabled by default)
- Increase `num_workers` for faster data loading
- Monitor PDE loss weight λ - crucial for performance
- Use appropriate spatial/temporal downsampling ratios

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original CDAnet paper authors for the theoretical foundation
- PyTorch team for the deep learning framework
- Scientific computing community for numerical methods

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/CDAsmoother/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/CDAsmoother/discussions)
- **Email**: your.email@domain.com

---

**⭐ Star this repository if you find it useful!**
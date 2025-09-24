# CDAsmoother: Physics-Informed Neural Networks for Fluid Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready implementation of **CDAnet** (Continuous Data Assimilation Network), a physics-informed deep neural network for high-resolution reconstruction of Rayleigh-Bénard convection from sparse observations.

Based on: *"CDAnet: A Physics-Informed Deep Neural Network for Downscaling Fluid Flows"* by Hammoud et al. (2022)

## 🚀 Quick Start (Production Ready)

### Single Command Training
```bash
# Complete end-to-end training with paper-exact configuration
python train_paper_config.py
```

**That's it!** This command provides:
- ✅ **Paper-exact hyperparameters** (λ=0.01, lr=0.1, 3000 PDE points)
- ✅ **Automatic data generation** if not found
- ✅ **Fixed loss function** (removed harmful clamping)
- ✅ **Proper normalization** handling
- ✅ **GPU optimization** with mixed precision
- ✅ **Comprehensive logging** and checkpointing
- ✅ **Production-ready architecture**

### View Results
```bash
# Create publication-quality visualizations
python visualize_results.py --checkpoint checkpoints/paper_config_final.pth --Ra 1e5
```

## 📋 What's New (Production Version)

### 🔧 Fixed Issues
- **Removed loss clamping** that prevented learning
- **Fixed data normalization** consistency between training/inference
- **Corrected PDE loss** implementation
- **Optimized hyperparameters** to match original paper

### 🏗️ Architecture (Paper-Compliant)
1. **3D U-Net Feature Extractor**
   - Inception-ResNet blocks with [1×1×1], [5×5×3], [9×9×5] kernels
   - Encoder-decoder with skip connections
   - 256 feature channels output

2. **Physics-Informed MLP**
   - Softplus activation (infinitely differentiable)
   - Coordinate-based prediction with automatic differentiation
   - 3 hidden layers [512, 512, 512] neurons

3. **Combined Loss Function**
   ```
   L_total = L_regression + λ × L_PDE
   where λ ∈ {0.001, 0.01, 0.1} (paper range)
   ```

## 📁 Project Structure

```
CDAsmoother/
├── cdanet/                          # Core CDAnet package
│   ├── models/                      # Neural network architectures
│   │   ├── cdanet.py               # Complete CDAnet model
│   │   ├── unet3d.py               # 3D U-Net with Inception-ResNet
│   │   ├── mlp.py                  # Physics-informed MLP
│   │   └── inception_resnet.py     # Paper-exact Inception blocks
│   ├── data/                        # Data loading and preprocessing
│   │   ├── dataset.py              # RB dataset with normalization
│   │   └── data_loader.py          # DataModule with 3000 PDE points
│   ├── training/                    # Training utilities
│   │   ├── trainer.py              # Main training loop (fixed)
│   │   └── losses.py               # Combined loss (fixed clamping)
│   ├── config/                      # Configuration management
│   │   └── config.py               # Experiment configurations
│   └── utils/                       # Utilities and visualization
├── train_paper_config.py            # 🆕 Production training script
├── train_cdanet.py                  # Alternative training script
├── visualize_results.py             # Result visualization (fixed)
├── evaluate_cdanet.py               # Model evaluation
├── rb_simulation.py                 # RB data generator
├── convert_rb_data.py               # Data format converter
└── requirements.txt                 # Dependencies
```

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **CUDA-capable GPU** (8GB+ recommended)
- **PyTorch with CUDA support**

### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd CDAsmoother

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## 📊 Training Options

### 1. Paper Configuration (Recommended)
```bash
# Use exact paper settings
python train_paper_config.py
```

**Paper hyperparameters:**
- **Learning rate**: 0.1 (range: 0.01-0.25)
- **PDE weight (λ)**: 0.01 (range: 0.001-0.1)
- **PDE points**: 3000 (exact paper setting)
- **Loss norms**: L2 regression + L2 PDE
- **Optimizer**: Adam with ReduceLROnPlateau
- **Batch size**: 4 (GPU memory optimized)

### 2. Custom Configuration
```bash
# Adjust hyperparameters
python train_cdanet.py --Ra 1e6 --learning_rate 0.05 --lambda_pde 0.001 --num_epochs 100
```

### 3. Different Rayleigh Numbers
```bash
# Train on different Ra numbers (as in paper)
python train_paper_config.py  # Ra=1e5 (default)
# Edit train_paper_config.py to change Ra to 1e6, 1e7, etc.
```

## 📈 Expected Results

### Performance (Paper Validation)
| Ra Number | Downsampling | RRMSE Temperature | RRMSE Velocity |
|-----------|--------------|------------------|----------------|
| 10^5      | (4, 4)       | ~1%              | ~2%            |
| 10^6      | (4, 4)       | ~1.5%            | ~3%            |
| 10^7      | (4, 4)       | ~2%              | ~4%            |

### Key Features
- **Physics-informed training** with RB governing equations
- **Multi-scale downscaling** from coarse to fine resolution
- **Temporal consistency** across time evolution
- **Cross-Ra generalization** capability

## 🔧 Data Generation

### Quick Data Generation (Recommended)
```bash
# Generate improved RB simulation data with physics-based validation
python generate_rb_data.py --Ra 1e5 --n_runs 5 --n_samples 50 --visualize
```

**Features:**
- ✅ **Numerically stable** Rayleigh-Bénard simulation
- ✅ **Multi-scale convection** patterns (large/medium/small scale)
- ✅ **Automatic consolidation** into CDAnet-compatible format
- ✅ **Built-in visualization** for data validation
- ✅ **Physics validation** with proper ranges and statistics

### Advanced Data Generation
```bash
# Different Rayleigh numbers
python generate_rb_data.py --Ra 1e6 --n_runs 10 --n_samples 100
python generate_rb_data.py --Ra 1e7 --n_runs 5 --n_samples 75

# Custom parameters
python generate_rb_data.py --Ra 1e5 --n_runs 3 --n_samples 25 --no-visualize
```

### Legacy Data Generation
Training automatically generates data if missing. Manual generation:

```bash
# Generate RB simulation data (older method)
python rb_simulation.py --Ra 1e5 --n_runs 10
python rb_simulation.py --Ra 1e6 --n_runs 10

# Convert to consolidated format
python convert_rb_data.py
```

## 📊 Evaluation & Visualization

### Model Evaluation
```bash
# Comprehensive evaluation with metrics
python evaluate_cdanet.py --checkpoint checkpoints/paper_config_final.pth
```

### Create Visualizations
```bash
# Generate paper-style figures
python visualize_results.py \
    --checkpoint checkpoints/paper_config_final.pth \
    --Ra 1e5 \
    --variable T \
    --output_dir ./results
```

**Output visualizations:**
- Comparison plots (Input vs Truth vs Prediction)
- Temporal evolution analysis
- Error field analysis
- Cross-sections and statistics

## ⚙️ Advanced Configuration

### Custom Training Loop
```python
from cdanet import CDAnet, RBDataModule, CDAnetTrainer
from cdanet.config import ExperimentConfig
from cdanet.utils import Logger

# Paper-exact configuration
config = ExperimentConfig()
config.loss.lambda_pde = 0.01          # Paper setting
config.optimizer.learning_rate = 0.1   # Paper range
config.data.pde_points = 3000           # Paper setting

# Setup components
model = CDAnet(**config.model.__dict__)
data_module = RBDataModule(**config.data.__dict__)
data_module.setup([1e5])

logger = Logger(log_dir='./logs', experiment_name='custom')
trainer = CDAnetTrainer(config, model, data_module, logger)

# Train with paper configuration
trainer.train()
```

## 🧮 Physics Implementation

### Rayleigh-Bénard Equations
```
1. Continuity:    ∇ · u = 0
2. Momentum-x:    ∂u/∂t + u·∇u = -∂p/∂x + Pr∇²u
3. Momentum-y:    ∂v/∂t + u·∇v = -∂p/∂y + Pr∇²v + RaPrT
4. Energy:        ∂T/∂t + u·∇T = ∇²T
```

### PDE Loss Computation
```python
# Automatic differentiation for PDE residuals
def compute_pde_loss(self, predictions, derivatives):
    # Extract variables: T, p, u, v = predictions[..., 0:4]
    # Compute residuals for all 4 governing equations
    # Return L1/L2 norm of combined residuals
```

## 🚨 Important Notes

### Production Fixes Applied
1. **Removed harmful loss clamping** that prevented convergence
2. **Fixed normalization consistency** between training/inference
3. **Set paper-exact hyperparameters** for reproducibility
4. **Optimized numerical stability** in PDE loss computation

### Troubleshooting
- **Constant predictions**: Fixed by removing loss clamping
- **Color saturation in plots**: Fixed by proper denormalization
- **Training instability**: Fixed with paper hyperparameters
- **Memory issues**: Use smaller batch_size (default: 4)

## 📚 Citation

If you use this code, please cite the original paper:
```bibtex
@article{hammoud2022cdanet,
  title={CDAnet: A Physics-Informed Deep Neural Network for Downscaling Fluid Flows},
  author={Hammoud, Mohamad Abed El Rahman and Titi, Edriss S and Hoteit, Ibrahim and Knio, Omar},
  journal={Journal of Advances in Modeling Earth Systems},
  year={2022},
  publisher={Wiley}
}
```

## 🤝 Contributing

This is a production-ready implementation. For issues or improvements, please submit pull requests with:
- Clear description of changes
- Test results on Ra=1e5 dataset
- Validation against paper benchmarks

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**⭐ Star this repository if you find it useful for your research!**
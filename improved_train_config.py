#!/usr/bin/env python3
"""
Improved training configuration for CDAnet with better hyperparameters.
Addresses the issues of insufficient training data, poor convergence, and flat predictions.
"""

import os
import torch
from datetime import datetime

from cdanet.models import CDAnet
from cdanet.config import ExperimentConfig
from cdanet.data import RBDataModule
from cdanet.training import CDAnetTrainer
from cdanet.utils import Logger


def create_improved_config():
    """Create improved configuration with better hyperparameters for stable training."""
    config = ExperimentConfig()

    # Experiment info
    config.experiment_name = f"cdanet_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.description = "CDAnet training with improved hyperparameters for better convergence"

    # 🔧 DATA IMPROVEMENTS: Generate more training data
    config.data.data_dir = os.path.abspath('./rb_data_numerical')
    config.data.spatial_downsample = 4
    config.data.temporal_downsample = 4
    config.data.clip_length = 8
    config.data.Ra_numbers = [1e5]
    config.data.batch_size = 2  # Smaller batch for stable training
    config.data.num_workers = 2  # Reduce to avoid memory issues
    config.data.pde_points = 4096  # More PDE points for better physics constraint
    config.data.normalize = True

    # 🔧 MODEL IMPROVEMENTS: Smaller model for limited data
    config.model.in_channels = 4
    config.model.feature_channels = 128  # Reduce from 256 to prevent overfitting
    config.model.base_channels = 32      # Add base channels parameter
    config.model.mlp_hidden_dims = [256, 256]  # Reduce from [512, 512, 512]
    config.model.activation = 'relu'     # Change to ReLU for faster training
    config.model.coord_dim = 3
    config.model.output_dim = 4

    # 🔧 LOSS IMPROVEMENTS: Better physics loss balance
    config.loss.lambda_pde = 0.001  # Reduce PDE loss weight for initial training
    config.loss.regression_norm = 'l2'
    config.loss.pde_norm = 'l2'
    config.loss.Ra = 1e5
    config.loss.Pr = 0.7
    config.loss.Lx = 3.0
    config.loss.Ly = 1.0

    # 🔧 OPTIMIZER IMPROVEMENTS: More conservative settings
    config.optimizer.optimizer_type = 'adam'
    config.optimizer.learning_rate = 0.001  # Much lower LR for stability
    config.optimizer.weight_decay = 1e-5    # Lower weight decay
    config.optimizer.grad_clip_max_norm = 0.5  # Tighter gradient clipping
    config.optimizer.scheduler_type = 'cosine'  # Cosine scheduler for smooth decay
    config.optimizer.T_max = 1000  # Cosine period
    config.optimizer.eta_min = 1e-6

    # 🔧 TRAINING IMPROVEMENTS: More epochs and patience
    config.training.num_epochs = 1000  # Much more epochs for convergence
    config.training.clips_per_epoch = -1  # Use all available data
    config.training.val_interval = 25     # Validate every 25 epochs
    config.training.checkpoint_interval = 100
    config.training.save_best = True
    config.training.early_stopping = True
    config.training.patience = 200  # More patience for slow convergence
    config.training.min_delta = 1e-6
    config.training.use_amp = False  # Disable AMP for numerical stability
    config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.training.log_interval = 5   # More frequent logging
    config.training.output_dir = './outputs'
    config.training.checkpoint_dir = './checkpoints'
    config.training.log_dir = './logs'

    return config


def generate_more_training_data(data_dir: str, Ra: float = 1e5):
    """Generate more training data for better model training."""
    print("🔄 Generating additional training data...")

    # Create data module for synthetic data generation
    temp_data_module = RBDataModule(
        data_dir=data_dir,
        spatial_downsample=4,
        temporal_downsample=4,
        clip_length=8,
        batch_size=1,
        normalize=True
    )

    # Generate multiple data files for more variety
    data_files = []
    for i in range(3):  # Generate 3 different datasets
        output_file = os.path.join(data_dir, f'rb_data_Ra_{Ra:.0e}_run_{i+1}.h5')
        if not os.path.exists(output_file):
            print(f"  Generating run {i+1}/3...")
            synthetic_file = temp_data_module.create_synthetic_data(
                output_path=output_file,
                Ra=Ra,
                nx=512,   # Higher resolution for more detail
                ny=128,   # Higher resolution for more detail
                nt=4000,  # More timesteps = more clips
                seed=42 + i  # Different seeds for variety
            )
            data_files.append(synthetic_file)
            print(f"    ✅ Generated: {os.path.basename(synthetic_file)}")
        else:
            data_files.append(output_file)
            print(f"    ✅ Exists: {os.path.basename(output_file)}")

    return data_files


def main():
    print("=" * 60)
    print("CDAnet Improved Training Configuration")
    print("=" * 60)

    # Create improved configuration
    config = create_improved_config()

    print(f"Experiment: {config.experiment_name}")
    print(f"Learning Rate: {config.optimizer.learning_rate}")
    print(f"Lambda PDE: {config.loss.lambda_pde}")
    print(f"Model Features: {config.model.feature_channels}")
    print(f"MLP Dims: {config.model.mlp_hidden_dims}")
    print(f"Device: {config.training.device}")
    print("=" * 60)

    # Generate more training data
    os.makedirs(config.data.data_dir, exist_ok=True)
    data_files = generate_more_training_data(config.data.data_dir, config.data.Ra_numbers[0])

    print(f"✅ Training data ready: {len(data_files)} files")

    # Setup data module
    print("Setting up data module...")
    data_module = RBDataModule(
        data_dir=config.data.data_dir,
        spatial_downsample=config.data.spatial_downsample,
        temporal_downsample=config.data.temporal_downsample,
        clip_length=config.data.clip_length,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pde_points=config.data.pde_points,
        normalize=config.data.normalize
    )

    try:
        data_module.setup(config.data.Ra_numbers)
        print("✅ Data module setup successful")

        # Print improved data info
        data_info = data_module.get_dataset_info()
        total_samples = sum(info['num_samples'] for info in data_info.values())
        print(f"📊 Total training samples: {total_samples}")

        for key, info in data_info.items():
            print(f"  {key}: {info['num_samples']} samples, shape {info['high_res_shape']}")

        if total_samples < 50:
            print("⚠️  Warning: Still limited training data. Consider generating more runs.")

    except Exception as e:
        print(f"❌ Data module setup failed: {e}")
        raise

    # Create smaller, more appropriate model
    print("Creating improved model...")
    model = CDAnet(
        in_channels=config.model.in_channels,
        feature_channels=config.model.feature_channels,
        base_channels=config.model.base_channels,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        activation=config.model.activation,
        coord_dim=config.model.coord_dim,
        output_dim=config.model.output_dim
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB")

    # Setup logger
    print("Setting up logger...")
    logger = Logger(
        log_dir=config.training.log_dir,
        experiment_name=config.experiment_name,
        use_tensorboard=True,
        use_wandb=False,
        config=config.to_dict()
    )

    # Create trainer
    print("Creating trainer...")
    trainer = CDAnetTrainer(
        config=config,
        model=model,
        data_module=data_module,
        logger=logger
    )

    # Start improved training
    print("🚀 Starting improved training...")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Early stopping patience: {config.training.patience}")
    print(f"  Validation interval: {config.training.val_interval}")
    print("=" * 60)

    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        logger.close()

    # Save final model
    final_checkpoint = os.path.join(config.training.checkpoint_dir, 'improved_model_final.pth')
    try:
        trainer._save_checkpoint('improved_model_final.pth', trainer.current_epoch, {})
        print(f"💾 Final model saved: {final_checkpoint}")
    except Exception as e:
        print(f"⚠️  Failed to save final checkpoint: {e}")

    print("=" * 60)
    print("🎯 Training Summary:")
    print(f"  Final epoch: {trainer.current_epoch}")
    print(f"  Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"  Model checkpoint: {final_checkpoint}")
    print("=" * 60)


if __name__ == '__main__':
    main()
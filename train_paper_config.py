#!/usr/bin/env python3
"""
Training script with paper-exact configuration for CDAnet.
Uses the exact hyperparameters from the original paper.
"""

import os
import torch
from datetime import datetime

from cdanet.models import CDAnet
from cdanet.config import ExperimentConfig
from cdanet.data import RBDataModule
from cdanet.training import CDAnetTrainer
from cdanet.utils import Logger


def create_paper_config():
    """Create experiment config matching the original paper."""
    config = ExperimentConfig()

    # Experiment info
    config.experiment_name = f"cdanet_paper_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.description = "CDAnet training with exact paper configuration"

    # Data configuration (paper settings)
    config.data.data_dir = './rb_data_numerical'
    config.data.spatial_downsample = 4
    config.data.temporal_downsample = 4
    config.data.clip_length = 8
    config.data.Ra_numbers = [1e5]  # Start with Ra=10^5 as in paper
    config.data.batch_size = 4  # Small batch for stability
    config.data.num_workers = 4
    config.data.pde_points = 3000  # Exact paper setting
    config.data.normalize = True  # Paper uses normalization

    # Model configuration (paper architecture)
    config.model.in_channels = 4
    config.model.feature_channels = 256  # Standard U-Net features
    config.model.mlp_hidden_dims = [512, 512, 512]  # Deep MLP as in paper
    config.model.activation = 'softplus'  # Paper specifies softplus
    config.model.coord_dim = 3
    config.model.output_dim = 4

    # Loss configuration (paper settings)
    config.loss.lambda_pde = 0.01  # Paper range: 0.001-0.1, start with 0.01
    config.loss.regression_norm = 'l2'  # L2 norm as typical in paper
    config.loss.pde_norm = 'l2'
    config.loss.Ra = 1e5
    config.loss.Pr = 0.7
    config.loss.Lx = 3.0
    config.loss.Ly = 1.0

    # Optimizer configuration (paper settings)
    config.optimizer.optimizer_type = 'adam'  # Adam is more stable than SGD
    config.optimizer.learning_rate = 0.1  # Paper range: 0.01-0.25
    config.optimizer.weight_decay = 1e-4
    config.optimizer.scheduler_type = 'plateau'
    config.optimizer.patience = 10
    config.optimizer.factor = 0.5
    config.optimizer.min_lr = 1e-6

    # Training configuration
    config.training.num_epochs = 50  # Start with fewer epochs for testing
    config.training.clips_per_epoch = None  # Use all clips
    config.training.val_interval = 5
    config.training.checkpoint_interval = 10
    config.training.save_best = True
    config.training.early_stopping = True
    config.training.patience = 20
    config.training.min_delta = 1e-6
    config.training.use_amp = False  # Disable AMP for stability
    config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.training.log_interval = 10
    config.training.output_dir = './outputs'
    config.training.checkpoint_dir = './checkpoints'
    config.training.log_dir = './logs'

    return config


def main():
    print("=" * 60)
    print("CDAnet Training - Paper Configuration")
    print("=" * 60)

    # Create paper-exact configuration
    config = create_paper_config()

    print(f"Experiment: {config.experiment_name}")
    print(f"Lambda PDE: {config.loss.lambda_pde}")
    print(f"Learning Rate: {config.optimizer.learning_rate}")
    print(f"PDE Points: {config.data.pde_points}")
    print(f"Device: {config.training.device}")
    print("=" * 60)

    # Setup data
    print("Setting up data...")
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
    data_module.setup(config.data.Ra_numbers)

    # Print data info
    data_info = data_module.get_dataset_info()
    for key, info in data_info.items():
        print(f"{key}: {info['num_samples']} samples, shape {info['high_res_shape']}")

    # Create model
    print("Creating model...")
    model = CDAnet(
        in_channels=config.model.in_channels,
        feature_channels=config.model.feature_channels,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        activation=config.model.activation,
        coord_dim=config.model.coord_dim,
        output_dim=config.model.output_dim
    )
    model.print_model_info()

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

    # Start training
    print("Starting training with paper configuration...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        logger.close()

    # Save final model
    final_checkpoint = os.path.join(config.training.checkpoint_dir, 'paper_config_final.pth')
    trainer._save_checkpoint('paper_config_final.pth', trainer.current_epoch, {})
    print(f"Final model saved: {final_checkpoint}")


if __name__ == '__main__':
    main()
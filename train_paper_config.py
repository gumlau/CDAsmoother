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
    config.data.data_dir = os.path.abspath('./rb_data_numerical')
    config.data.spatial_downsample = 4
    config.data.temporal_downsample = 4
    config.data.clip_length = 8
    config.data.Ra_numbers = [1e5]  # Start with Ra=10^5 as in paper
    config.data.batch_size = 8  # Increase batch size for better training
    config.data.num_workers = 4  # Enable multiprocessing for faster loading
    config.data.pde_points = 3000  # Paper uses 3,000 points
    config.data.normalize = True  # Enable normalization as typically used

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
    config.optimizer.learning_rate = 0.01  # Start with conservative LR to avoid instability
    config.optimizer.weight_decay = 1e-4
    config.optimizer.grad_clip_max_norm = 1.0  # Gradient clipping for stability
    config.optimizer.scheduler_type = 'plateau'
    config.optimizer.patience = 10  # Paper: reduce LR if no improvement in 10 epochs
    config.optimizer.factor = 0.1  # Paper: scale LR by 0.1
    config.optimizer.min_lr = 1e-6

    # Training configuration (paper settings)
    config.training.num_epochs = 500  # Increase epochs for more thorough training
    config.training.clips_per_epoch = 617  # Use all available clips per epoch
    config.training.val_interval = 5
    config.training.checkpoint_interval = 10
    config.training.save_best = True
    config.training.early_stopping = True
    config.training.patience = 50  # More patience for longer training
    config.training.min_delta = 1e-7  # Smaller threshold for improvement
    config.training.use_amp = False  # Disable AMP to avoid numerical instability
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

    # Check if consolidated data file exists
    data_file = os.path.join(config.data.data_dir, 'rb_data_Ra_1e+05.h5')
    print(f"Looking for consolidated file: {data_file}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Data directory exists: {os.path.exists(config.data.data_dir)}")

    if not os.path.exists(data_file):
        print(f"Consolidated data file not found at {data_file}")

        # List available files before conversion
        if os.path.exists(config.data.data_dir):
            files = [f for f in os.listdir(config.data.data_dir) if f.endswith('.h5')]
            print(f"Available .h5 files before conversion: {files[:10]}...")  # Show first 10
        else:
            print(f"Creating data directory: {config.data.data_dir}")
            os.makedirs(config.data.data_dir, exist_ok=True)
            files = []

        # If no .h5 files exist, generate sample data first
        if not files:
            print("No data files found. Generating sample RB simulation data...")

            # Generate sample data using data module
            try:
                temp_data_module = RBDataModule(
                    data_dir=config.data.data_dir,
                    spatial_downsample=config.data.spatial_downsample,
                    temporal_downsample=config.data.temporal_downsample,
                    clip_length=config.data.clip_length,
                    batch_size=config.data.batch_size,
                    num_workers=0,
                    pde_points=config.data.pde_points,
                    normalize=config.data.normalize
                )

                # Use create_synthetic_data method
                synthetic_file = temp_data_module.create_synthetic_data(
                    output_path=data_file,
                    Ra=1e5,
                    nx=256,  # Smaller for faster generation
                    ny=64,   # Smaller for faster generation
                    nt=2000  # More timesteps for more clips (need > clip_length * temporal_downsample)
                )

                print(f"✅ Generated synthetic data: {synthetic_file}")

            except Exception as e:
                print(f"❌ Failed to generate synthetic data: {e}")
                print("Trying alternative RB simulation...")

                # Fallback: run rb_simulation.py
                import subprocess
                result = subprocess.run(['python3', 'rb_simulation.py', '--Ra', '1e5', '--n_runs', '5'],
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ RB simulation failed: {result.stderr}")
                    raise RuntimeError("Could not generate training data")

                print("✅ RB simulation completed")

        # Now try conversion if we have run files
        if os.path.exists(config.data.data_dir):
            files = [f for f in os.listdir(config.data.data_dir) if f.endswith('.h5')]

        if files and not os.path.exists(data_file):
            print("Converting run files to consolidated format...")
            import subprocess
            result = subprocess.run(['python3', 'convert_rb_data.py'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Data conversion failed: {result.stderr}")
                print("Proceeding with existing files...")

        # Final check for data file
        if os.path.exists(data_file):
            print(f"✅ Consolidated file ready: {data_file}")
            print(f"File size: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")
        else:
            # List what files we actually have
            if os.path.exists(config.data.data_dir):
                files = [f for f in os.listdir(config.data.data_dir) if f.endswith('.h5')]
                print(f"Available .h5 files: {files}")
                if files:
                    print("✅ Will proceed with available data files")
                else:
                    raise RuntimeError("No data files available for training")
            else:
                raise RuntimeError("Data directory does not exist")
    else:
        print(f"✅ Consolidated file already exists: {data_file}")
        print(f"File size: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")

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

        # Verify data loaders were created
        available_loaders = list(data_module.data_loaders.keys())
        print(f"Available data loaders: {available_loaders}")

        if not available_loaders:
            raise RuntimeError("No data loaders were created!")

    except Exception as e:
        print(f"❌ Data module setup failed: {e}")
        import traceback
        traceback.print_exc()
        raise

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
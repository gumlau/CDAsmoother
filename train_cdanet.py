#!/usr/bin/env python
"""
Main training script for CDAnet.

Usage:
    python train_cdanet.py --config config.yaml
    python train_cdanet.py --Ra 1e5 --spatial_downsample 4 --temporal_downsample 4
"""

import argparse
import os
import yaml
import torch
from datetime import datetime

# Import CDAnet components
from cdanet.models import CDAnet
from cdanet.config import ExperimentConfig, create_config_for_ra, PAPER_CONFIGS
from cdanet.data import RBDataModule
from cdanet.training import CDAnetTrainer
from cdanet.utils import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train CDAnet for Rayleigh-Bénard convection')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--preset', type=str, choices=list(PAPER_CONFIGS.keys()),
                       help='Use predefined configuration from paper')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./rb_data_numerical',
                       help='Directory containing data files')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--spatial_downsample', type=int, default=4, help='Spatial downsampling factor')
    parser.add_argument('--temporal_downsample', type=int, default=4, help='Temporal downsampling factor')
    
    # Training parameters (optimized for CUDA)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (optimized for GPU)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--lambda_pde', type=float, default=0.01, help='PDE loss weight')

    # Model parameters (optimized for GPU)
    parser.add_argument('--feature_channels', type=int, default=512, help='Feature channels in U-Net')
    parser.add_argument('--mlp_layers', type=int, default=4, help='Number of MLP hidden layers')
    parser.add_argument('--mlp_width', type=int, default=512, help='Width of MLP hidden layers')

    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--experiment_name', type=str, help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    # Hardware (defaults to CUDA)
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for training (defaults to CUDA)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    
    # Flags
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--no_data_gen', action='store_true', help='Disable automatic data generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ExperimentConfig()
    config.update_from_dict(config_dict)
    config.validate()
    
    return config


def create_config_from_args(args) -> ExperimentConfig:
    """Create configuration from command line arguments."""
    if args.preset:
        config = PAPER_CONFIGS[args.preset]
    else:
        config = create_config_for_ra(args.Ra, args.spatial_downsample, args.temporal_downsample)
    
    # Override with command line arguments
    config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    config.training.num_epochs = args.num_epochs
    config.training.output_dir = args.output_dir
    
    # Set device (default to CUDA, fallback if needed)
    if args.device == 'auto':
        if torch.cuda.is_available():
            config.training.device = 'cuda'
        elif torch.backends.mps.is_available():
            config.training.device = 'mps'
        else:
            config.training.device = 'cpu'
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            config.training.device = 'cuda'
        else:
            print("Warning: CUDA not available, falling back to CPU")
            config.training.device = 'cpu'
    else:
        config.training.device = args.device
    
    config.training.use_amp = not args.no_amp
    
    config.optimizer.learning_rate = args.learning_rate
    config.loss.lambda_pde = args.lambda_pde
    
    config.model.feature_channels = args.feature_channels
    config.model.mlp_hidden_dims = [args.mlp_width] * args.mlp_layers
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"cdanet_Ra_{args.Ra:.0e}_ds_{args.spatial_downsample}_{args.temporal_downsample}_{timestamp}"
    
    config.validate()
    return config


def setup_data(config: ExperimentConfig, generate_if_missing: bool = False) -> RBDataModule:
    """Setup data module and generate data if needed."""
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
    
    # Check if data exists
    Ra = config.data.Ra_numbers[0]
    data_file = data_module._find_data_file(Ra)
    
    if data_file is None and generate_if_missing:
        print(f"Data file not found for Ra={Ra}, generating RB simulation data...")
        print("This may take a few minutes...")
        # Use rb_simulation.py to generate real data
        import subprocess
        result = subprocess.run([
            'python3', 'rb_simulation.py',
            '--Ra', str(Ra),
            '--n_runs', '10'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Warning: RB simulation failed. Generating synthetic data as fallback...")
            data_file = os.path.join(config.data.data_dir, f'rb_data_Ra_{Ra:.0e}.h5')
            data_module.create_synthetic_data(data_file, Ra=Ra)
        else:
            print("RB simulation data generated successfully!")
            # Convert to consolidated format
            subprocess.run(['python3', 'convert_rb_data.py'], capture_output=True)
    
    # Setup data module
    data_module.setup(config.data.Ra_numbers)
    
    return data_module


def main():
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = create_config_from_args(args)
    
    print("=" * 60)
    print("CDAnet Training Configuration")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Rayleigh number: {config.data.Ra_numbers[0]:.0e}")
    print(f"Downsampling: γ_s={config.data.spatial_downsample}, γ_t={config.data.temporal_downsample}")
    print(f"Device: {config.training.device}")
    print(f"Output directory: {config.training.output_dir}")
    print("=" * 60)
    
    # Setup data (auto-generate by default)
    print("Setting up data...")
    generate_data = not args.no_data_gen  # Generate by default unless disabled
    data_module = setup_data(config, generate_if_missing=generate_data)
    
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
        use_tensorboard=config.training.tensorboard,
        use_wandb=config.training.wandb,
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
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
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


if __name__ == '__main__':
    main()
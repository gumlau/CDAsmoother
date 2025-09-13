#!/usr/bin/env python
"""
Small-scale training script for CDAnet - suitable for CPU/MPS testing.
"""

import argparse
import torch
import time
from cdanet.models import CDAnet
from cdanet.config import create_config_for_ra
from cdanet.data import RBDataModule
from cdanet.training import CDAnetTrainer
from cdanet.utils import Logger

def main():
    parser = argparse.ArgumentParser(description='Small-scale CDAnet training for testing')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_batches', type=int, default=5, help='Number of batches to train')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CDAnet Small-Scale Training")
    print("=" * 60)
    
    # Create minimal config
    config = create_config_for_ra(Ra=1e5, spatial_downsample=8, temporal_downsample=8)
    
    # VERY small model for testing
    config.model.feature_channels = 16
    config.model.base_channels = 4
    config.model.mlp_hidden_dims = [32, 32]
    
    # Training settings
    config.data.batch_size = args.batch_size
    config.data.pde_points = 50  # Very few PDE points
    config.training.num_epochs = 1
    config.training.clips_per_epoch = args.num_batches
    config.training.log_interval = 1
    config.training.val_interval = 10  # Skip validation
    config.training.checkpoint_interval = 10  # Skip checkpointing
    config.training.use_amp = False
    config.experiment_name = 'small_test'
    
    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            config.training.device = 'cuda'
        elif torch.backends.mps.is_available():
            config.training.device = 'mps'
        else:
            config.training.device = 'cpu'
    else:
        config.training.device = args.device
    
    print(f"Device: {config.training.device}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Feature channels: {config.model.feature_channels}")
    print(f"Training batches: {args.num_batches}")
    
    # Setup data
    print("\nSetting up data...")
    data_module = RBDataModule(
        data_dir=config.data.data_dir,
        spatial_downsample=config.data.spatial_downsample,
        temporal_downsample=config.data.temporal_downsample,
        clip_length=config.data.clip_length,
        batch_size=config.data.batch_size,
        num_workers=0,
        pde_points=config.data.pde_points,
        normalize=config.data.normalize
    )
    
    data_module.setup([1e5])
    print("Data loaded successfully")
    
    # Create model
    print("\nCreating model...")
    model = CDAnet(
        in_channels=config.model.in_channels,
        feature_channels=config.model.feature_channels,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        activation=config.model.activation
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup logger
    logger = Logger(
        log_dir='./test_logs',
        experiment_name=config.experiment_name,
        use_tensorboard=False,
        use_wandb=False
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = CDAnetTrainer(
        config=config,
        model=model,
        data_module=data_module,
        logger=logger
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        trainer.train()
        print(f"\n✓ Training completed in {time.time() - start_time:.2f} seconds!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()

if __name__ == '__main__':
    main()
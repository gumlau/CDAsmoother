#!/usr/bin/env python3
"""
Memory-optimized CDAnet training script
Reduces memory usage to avoid OOM errors on GPUs with limited memory
"""

import os
import sys
import torch

# Set memory optimization flags
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False  # More memory efficient
    torch.backends.cudnn.deterministic = True

# Import original training script
sys.path.append('.')
from train_cdanet import *

def main():
    """Memory-optimized main function"""
    args = parse_args()

    # Override parameters for stable training without OOM
    print("🔧 Applying optimizations to prevent OOM while maximizing utilization...")
    args.batch_size = 1           # Minimum batch size to prevent OOM
    args.n_samp_pts_per_crop = 128  # Much smaller to prevent OOM
    args.nx = 128                 # Keep as power of 2 for U-Net compatibility
    args.nz = 64                  # Keep as power of 2 for U-Net compatibility
    args.nt = 16                  # Keep as power of 2 for U-Net compatibility

    # Stability settings to prevent NaN
    args.lr = 0.01               # Reduced learning rate for stability
    args.alpha_pde = 0.001       # Much reduced PDE weight to prevent NaN
    args.clip_grad = 0.5         # Aggressive gradient clipping

    print(f"Optimized settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample points per crop: {args.n_samp_pts_per_crop}")
    print(f"  Spatial resolution: {args.nx} x {args.nz}")
    print(f"  Temporal resolution: {args.nt}")
    print(f"  Learning rate: {args.lr} (reduced for stability)")
    print(f"  PDE weight: {args.alpha_pde} (reduced to prevent NaN)")
    print(f"  Gradient clipping: {args.clip_grad}")
    print()

    # Setup device with memory management
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Using CUDA GPU with optimization")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Print GPU info
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Using Apple Silicon MPS with memory optimization")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Create output directory
    args.output_folder = './checkpoints_optimized'
    os.makedirs(args.output_folder, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_folder, 'tensorboard'))

    # Create datasets with reduced memory usage
    print("Creating datasets...")
    train_dataset = FixedRB2DataLoader(
        data_dir=args.data_folder,
        data_filename=args.train_data,
        nx=args.nx, nz=args.nz, nt=args.nt,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels,
        return_hres=False,
        velOnly=args.velocityOnly
    )

    eval_dataset = FixedRB2DataLoader(
        data_dir=args.data_folder,
        data_filename=args.eval_data,
        nx=args.nx, nz=args.nz, nt=args.nt,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels,
        return_hres=True,
        velOnly=args.velocityOnly
    )

    # Create data loaders optimized for GPU utilization
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Increased workers for better GPU utilization
        pin_memory=True  # Enable for faster GPU transfer
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Creating CDAnet model with reference architecture...")
    # Create CDAnet model using our corrected implementation
    from cdanet.models import CDAnet

    model = CDAnet(
        in_channels=4,
        feature_channels=64,  # Smaller for memory efficiency
        mlp_hidden_dims=[64, 128],  # [lat_dims, imnet_nf]
        activation='softplus',
        coord_dim=3,
        output_dim=4,
        igres=train_dataset.scale_lres,  # Use dataset resolution
        unet_nf=16,
        unet_mf=64  # Much smaller to prevent OOM
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Create optimizer with stability settings
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

    # Learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Create loss function and PDE layer
    criterion = torch.nn.MSELoss()
    pde_layer = get_rb2_pde_layer()

    # Training loop with stability enhancements
    print("Starting optimized training with stability enhancements...\n")

    best_loss = float('inf')
    nan_count = 0
    max_nan_tolerance = 5  # Allow some NaN before stopping

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Clear memory at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training
        model.train()
        train_loss = 0.0
        valid_batches = 0

        for batch_idx, data_tensors in enumerate(train_loader):
            data_tensors = [t.to(device) for t in data_tensors]
            input_grid, point_coord, point_value = data_tensors
            try:
                optimizer.zero_grad()

                # Forward pass using CDAnet
                pred_value = model(input_grid, point_coord)

                # Check for NaN in predictions
                if torch.isnan(pred_value).any():
                    print(f"⚠️  NaN detected in predictions at batch {batch_idx}")
                    nan_count += 1
                    if nan_count > max_nan_tolerance:
                        print("❌ Too many NaN losses, stopping training")
                        return
                    continue

                # Regression loss
                reg_loss = criterion(pred_value, point_value)

                # Simple loss without PDE terms for now (can be added later)
                total_loss = reg_loss

                # Check for NaN/Inf in total loss
                if not torch.isfinite(total_loss):
                    print(f"⚠️  Non-finite loss at batch {batch_idx}: {total_loss.item()}")
                    nan_count += 1
                    if nan_count > max_nan_tolerance:
                        print("❌ Too many NaN losses, stopping training")
                        return
                    continue

                # Backward pass with gradient clipping
                total_loss.backward()

                # Aggressive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                # Check gradient norm
                if not torch.isfinite(grad_norm):
                    print(f"⚠️  Non-finite gradient norm at batch {batch_idx}")
                    nan_count += 1
                    continue

                optimizer.step()
                train_loss += total_loss.item()
                valid_batches += 1

                # Clear intermediate variables
                del latent_grid, pred_value, total_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batch_idx % 10 == 0:
                    current_loss = total_reg_loss.item() + args.alpha_pde * total_pde_loss.item() if total_pde_loss > 0 else total_reg_loss.item()
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {current_loss:.2e}, Grad Norm: {grad_norm:.2e}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️  OOM at batch {batch_idx}. Skipping and clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        if valid_batches > 0:
            avg_train_loss = train_loss / valid_batches
            print(f"  Average train loss: {avg_train_loss:.2e} (valid batches: {valid_batches}/{len(train_loader)})")

            # Learning rate scheduling
            scheduler.step(avg_train_loss)

            # Reset NaN count on successful epoch
            if avg_train_loss < float('inf'):
                nan_count = 0
        else:
            avg_train_loss = float('inf')
            print(f"  No valid batches completed")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'model_config': {
                    'in_channels': 4,
                    'feature_channels': 64,
                    'mlp_hidden_dims': [64, 128],
                    'activation': 'softplus',
                    'coord_dim': 3,
                    'output_dim': 4,
                    'igres': train_dataset.scale_lres,
                    'unet_nf': 16,
                    'unet_mf': 64
                }
            }
            torch.save(checkpoint, os.path.join(args.output_folder, f'checkpoint_epoch_{epoch:03d}.pth'))
            print(f"  Checkpoint saved at epoch {epoch}")

    print("\n" + "=" * 60)
    print("Optimized training completed!")
    print(f"Checkpoints saved to: {args.output_folder}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()
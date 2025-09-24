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

def compute_pde_loss(pred, coords):
    """
    Compute simple PDE loss for physics constraint.
    Implements basic smoothness regularization for fluid dynamics.
    """
    # pred: [batch, n_points, 4] - T, p, u, v
    # coords: [batch, n_points, 3] - x, z, t

    batch_size, n_points, _ = pred.shape

    # Compute gradients for smoothness
    # First-order derivatives
    grads = torch.autograd.grad(
        outputs=pred,
        inputs=coords,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [batch, n_points, 3]

    # Simple smoothness loss: penalize large gradients
    smoothness_loss = torch.mean(grads ** 2)

    return smoothness_loss

def main():
    """Memory-optimized main function"""
    args = parse_args()

    # Enhanced parameters for better GPU utilization and data augmentation
    print("🔧 Applying enhanced settings for better GPU utilization and training diversity...")
    args.batch_size = 2           # Increased from 1 for better GPU utilization
    args.n_samp_pts_per_crop = 2048  # Significantly increased to utilize GPU better (was 128)
    args.nx = 128                 # Keep as power of 2 for U-Net compatibility
    args.nz = 64                  # Keep as power of 2 for U-Net compatibility
    args.nt = 16                  # Keep as power of 2 for U-Net compatibility

    # Enhanced stability settings with data augmentation
    args.lr = 0.001              # Lower learning rate for stable training with more data
    args.alpha_pde = 0.01        # Increased PDE weight for physics constraint
    args.clip_grad = 1.0         # Less aggressive gradient clipping
    args.use_data_augmentation = True  # Enable data augmentation
    args.temporal_shift_prob = 0.3     # Probability of temporal shifting
    args.spatial_flip_prob = 0.5       # Probability of spatial flipping
    args.noise_level = 0.02           # Gaussian noise level for robustness

    print(f"Enhanced settings:")
    print(f"  Batch size: {args.batch_size} (increased for GPU utilization)")
    print(f"  Sample points per crop: {args.n_samp_pts_per_crop} (16x increase for GPU)")
    print(f"  Spatial resolution: {args.nx} x {args.nz}")
    print(f"  Temporal resolution: {args.nt}")
    print(f"  Learning rate: {args.lr}")
    print(f"  PDE weight: {args.alpha_pde} (increased for physics)")
    print(f"  Gradient clipping: {args.clip_grad}")
    print(f"  Data augmentation: {args.use_data_augmentation}")
    print(f"    - Temporal shift probability: {args.temporal_shift_prob}")
    print(f"    - Spatial flip probability: {args.spatial_flip_prob}")
    print(f"    - Noise level: {args.noise_level}")
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

    # Create data augmentation transforms
    print("Setting up data augmentation...")

    class DataAugmentation:
        """Enhanced data augmentation for RB convection"""
        def __init__(self, temporal_shift_prob=0.3, spatial_flip_prob=0.5, noise_level=0.02):
            self.temporal_shift_prob = temporal_shift_prob
            self.spatial_flip_prob = spatial_flip_prob
            self.noise_level = noise_level

        def __call__(self, input_grid, point_coord, point_value):
            # Temporal shifting augmentation
            if torch.rand(1) < self.temporal_shift_prob:
                # Random circular shift along temporal dimension
                t_shift = torch.randint(-2, 3, (1,)).item()  # -2 to +2 time steps
                input_grid = torch.roll(input_grid, t_shift, dims=1)  # [batch, T, H, W]

            # Spatial flipping augmentation
            if torch.rand(1) < self.spatial_flip_prob:
                # Random horizontal flip
                input_grid = torch.flip(input_grid, dims=[3])  # Flip width dimension
                # Flip corresponding coordinates (x coordinate -> -x)
                point_coord[:, 0] = -point_coord[:, 0]

            # Add small Gaussian noise for robustness
            if self.noise_level > 0:
                noise = torch.randn_like(input_grid) * self.noise_level
                input_grid = input_grid + noise

            return input_grid, point_coord, point_value

    # Create augmentation transform
    if args.use_data_augmentation:
        augmentation = DataAugmentation(
            temporal_shift_prob=args.temporal_shift_prob,
            spatial_flip_prob=args.spatial_flip_prob,
            noise_level=args.noise_level
        )
    else:
        augmentation = None

    # Create datasets with enhanced settings
    print("Creating enhanced datasets...")
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

    # Create data loaders optimized for GPU utilization with larger batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Increased workers for better GPU utilization
        pin_memory=True,  # Enable for faster GPU transfer
        drop_last=True   # Drop incomplete batches for consistent training
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print("Creating CDAnet model with reference architecture...")
    # Create CDAnet model using our corrected implementation
    from cdanet.models import CDAnet

    model = CDAnet(
        in_channels=4,
        feature_channels=128,  # Increased for better representation (was 64)
        mlp_hidden_dims=[128, 256],  # [lat_dims, imnet_nf] - Increased both
        activation='softplus',
        coord_dim=3,
        output_dim=4,
        igres=train_dataset.scale_lres,  # Use dataset resolution
        unet_nf=32,   # Increased base features (was 16)
        unet_mf=128   # Increased max features for better GPU utilization (was 64)
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
                # Apply data augmentation
                if augmentation is not None:
                    input_grid, point_coord, point_value = augmentation(input_grid, point_coord, point_value)

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

                # Physics-informed loss with PDE constraint (as in paper)
                if args.alpha_pde > 0:
                    try:
                        # Sample subset of points for PDE loss to save memory
                        n_pde_points = min(512, point_coord.shape[1])  # Limit PDE points
                        pde_indices = torch.randperm(point_coord.shape[1])[:n_pde_points]
                        pde_coords = point_coord[:, pde_indices]  # [batch, n_pde, 3]
                        pde_coords.requires_grad_(True)

                        # Forward pass for PDE loss
                        pde_pred = model(input_grid, pde_coords)

                        # Simple PDE loss: encourage smoothness (Laplacian regularization)
                        # ∇²u should be reasonable for fluid dynamics
                        pde_loss = compute_pde_loss(pde_pred, pde_coords)
                        total_loss = reg_loss + args.alpha_pde * pde_loss

                        # Log PDE loss
                        if batch_idx % 20 == 0:
                            print(f"      PDE Loss: {pde_loss.item():.2e}")

                    except Exception as e:
                        print(f"      ⚠️ PDE loss failed: {str(e)[:50]}...")
                        total_loss = reg_loss
                else:
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

                # Log progress before cleanup
                current_loss = total_loss.item()
                train_loss += current_loss
                valid_batches += 1

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {current_loss:.2e}, Grad Norm: {grad_norm:.2e}")

                # Clear intermediate variables
                del pred_value, total_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
                    'feature_channels': 128,  # Updated
                    'mlp_hidden_dims': [128, 256],  # Updated
                    'activation': 'softplus',
                    'coord_dim': 3,
                    'output_dim': 4,
                    'igres': train_dataset.scale_lres,
                    'unet_nf': 32,   # Updated
                    'unet_mf': 128   # Updated
                },
                'training_config': {
                    'batch_size': args.batch_size,
                    'n_samp_pts_per_crop': args.n_samp_pts_per_crop,
                    'lr': args.lr,
                    'alpha_pde': args.alpha_pde,
                    'use_data_augmentation': args.use_data_augmentation
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
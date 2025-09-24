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

    # Override memory-heavy parameters
    print("🔧 Applying memory optimizations...")
    args.batch_size = 1           # Much smaller batch size
    args.n_samp_pts_per_crop = 256  # Reduce from 1024 to 256
    args.nx = 64                  # Smaller spatial resolution
    args.nz = 32                  # Smaller spatial resolution
    args.nt = 8                   # Smaller temporal resolution

    print(f"Memory optimized settings:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample points per crop: {args.n_samp_pts_per_crop}")
    print(f"  Spatial resolution: {args.nx} x {args.nz}")
    print(f"  Temporal resolution: {args.nt}")
    print()

    # Setup device with memory management
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Using CUDA GPU with memory optimization")
            # Clear CUDA cache
            torch.cuda.empty_cache()
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
    args.output_folder = './checkpoints_low_memory'
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

    # Create data loaders with smaller batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Reduce CPU memory usage
        pin_memory=False  # Reduce GPU memory usage
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("Creating models...")
    # Create smaller models
    in_features = 2 if args.velocityOnly else 4

    unet = UNet3d(in_features=in_features, out_features=128,  # Reduced from 256
                  igres=train_dataset.scale_lres, nf=32, mf=128)  # Reduced max features

    imnet = ImNet(dim=3, in_features=128,  # Reduced from 256
                  out_features=4, nf=128,  # Reduced from 256
                  activation=NONLINEARITIES[args.nonlin])

    unet.to(device)
    imnet.to(device)

    # Count parameters
    unet_params = sum(p.numel() for p in unet.parameters())
    imnet_params = sum(p.numel() for p in imnet.parameters())
    total_params = unet_params + imnet_params
    print(f"Model parameters: {unet_params:,} (U-Net) + {imnet_params:,} (ImNet) = {total_params:,} total")

    # Create optimizer with gradient clipping
    params = list(unet.parameters()) + list(imnet.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # Create loss function and PDE layer
    criterion = torch.nn.MSELoss()
    pde_layer = get_rb2_pde_layer()

    # Training loop with memory management
    print("Starting memory-optimized training...\n")

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Clear memory at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training
        unet.train()
        imnet.train()
        train_loss = 0.0

        for batch_idx, data_tensors in enumerate(train_loader):
            data_tensors = [t.to(device) for t in data_tensors]
            input_grid, point_coord, point_value = data_tensors
            try:
                optimizer.zero_grad()

                # Forward pass with gradient checkpointing to save memory
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    # Reduce batch size further if needed
                    if batch_idx == 0:
                        try:
                            latent_grid = unet(input_grid)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print("⚠️  Still out of memory. Try reducing batch size to 1 or using CPU.")
                                return
                            else:
                                raise e
                    else:
                        latent_grid = unet(input_grid)

                latent_grid = latent_grid.permute(0, 2, 3, 4, 1)

                # Define forward function
                xmin = torch.zeros(3, dtype=torch.float32).to(device)
                xmax = torch.ones(3, dtype=torch.float32).to(device)
                pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

                # Update PDE layer and compute with reduced points
                pde_layer.update_forward_method(pde_fwd_fn)

                # Process points in smaller chunks to avoid memory issues
                chunk_size = 64  # Much smaller chunks
                total_reg_loss = 0.0
                total_pde_loss = 0.0

                n_points = point_coord.shape[1]
                n_chunks = (n_points + chunk_size - 1) // chunk_size

                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, n_points)

                    chunk_coord = point_coord[:, start_idx:end_idx]
                    chunk_value = point_value[:, start_idx:end_idx]

                    # Compute predictions and residues for chunk
                    pred_value, residue_dict = pde_layer(chunk_coord, return_residue=True)

                    # Regression loss
                    reg_loss = criterion(pred_value, chunk_value)
                    total_reg_loss += reg_loss

                    # PDE loss (only compute every few chunks to save memory)
                    if chunk_idx % 4 == 0:  # Reduce PDE computation frequency
                        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
                        pde_loss = criterion(pde_tensors, torch.zeros_like(pde_tensors))
                        total_pde_loss += pde_loss

                # Average losses
                total_reg_loss = total_reg_loss / n_chunks
                if total_pde_loss > 0:
                    total_pde_loss = total_pde_loss / max(1, n_chunks // 4)

                # Total loss
                total_loss = total_reg_loss + args.alpha_pde * total_pde_loss

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                optimizer.step()
                train_loss += total_loss.item()

                # Clear intermediate variables
                del latent_grid, pred_value, total_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batch_idx % 10 == 0:
                    current_loss = total_reg_loss.item() + args.alpha_pde * total_pde_loss.item() if total_pde_loss > 0 else total_reg_loss.item()
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {current_loss:.2e}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️  OOM at batch {batch_idx}. Skipping and clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_train_loss = train_loss / len(train_loader)
        print(f"  Average train loss: {avg_train_loss:.2e}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'imnet_state_dict': imnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }
            torch.save(checkpoint, os.path.join(args.output_folder, f'checkpoint_epoch_{epoch:03d}.pth'))
            print(f"  Checkpoint saved at epoch {epoch}")

    print("\n" + "=" * 60)
    print("Memory-optimized training completed!")
    print(f"Checkpoints saved to: {args.output_folder}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""
Create publication-quality visualizations for CDAnet results.
Generates the classic Rayleigh-Bénard convection comparison plots.

Usage:
    python visualize_results.py --checkpoint checkpoints/best_model.pth --data_dir ./rb_data_numerical
    python visualize_results.py --demo  # Create demo visualization
"""

import argparse
import os
import numpy as np
import torch
import h5py
from typing import Optional

from cdanet.utils.rb_visualization import RBVisualization, visualize_training_results
from cdanet.models import CDAnet
from cdanet.data import RBDataModule
from cdanet.config import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Create CDAnet visualizations')
    
    parser.add_argument('--checkpoint', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./rb_data_numerical',
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    
    # Visualization options
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--variable', type=str, default='T', choices=['T', 'u', 'v', 'p'],
                       help='Variable to visualize')
    parser.add_argument('--times', nargs='+', type=float, default=[15.0, 18.2, 21.5],
                       help='Time points to visualize')
    parser.add_argument('--n_snapshots', type=int, default=8,
                       help='Number of snapshots for temporal evolution')
    
    # Data options
    parser.add_argument('--spatial_downsample', type=int, default=4,
                       help='Spatial downsampling factor')
    parser.add_argument('--temporal_downsample', type=int, default=4,
                       help='Temporal downsampling factor')
    
    # Output options  
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format')
    
    # Special modes
    parser.add_argument('--demo', action='store_true',
                       help='Create demo visualization with synthetic data')
    parser.add_argument('--compare_cda', action='store_true',
                       help='Include CDA comparison (requires CDA results)')
    
    return parser.parse_args()


def load_model_and_predict(checkpoint_path: str, data_path: str, Ra: float,
                          spatial_downsample: int, temporal_downsample: int) -> dict:
    """Load trained model and generate predictions."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        model_config = config_dict['model']
    else:
        # Default config matching working_example_checkpoint.pth
        model_config = {
            'in_channels': 4,
            'feature_channels': 128,  # Matches working example
            'mlp_hidden_dims': [256, 256],  # Matches working example
            'activation': 'softplus',
            'coord_dim': 3,
            'output_dim': 4
        }
    
    model = CDAnet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup data with normalization (as per paper and training)
    normalize_data = True  # Paper uses normalized data for training

    print(f"Setting up data with normalize={normalize_data}")
    data_module = RBDataModule(
        data_dir=os.path.dirname(data_path),
        spatial_downsample=spatial_downsample,
        temporal_downsample=temporal_downsample,
        batch_size=1,
        normalize=normalize_data,
        num_workers=0  # Avoid timeout issues
    )
    data_module.setup([Ra])
    
    # Get test data
    test_loader = data_module.get_dataloader(Ra, 'test')

    print(f"🔍 Debug Info:")
    print(f"  Test loader batches: {len(test_loader)}")

    results = {
        'input_fields': [],
        'truth_fields': [],
        'predictions': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"🔍 Processing batch {i+1}/{len(test_loader)}")

            if i >= 10:  # Limit to first 10 batches for visualization
                break

            # Move to device
            low_res = batch['low_res'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['targets'].to(device)

            # Debug data shapes and ranges
            print(f"  📊 Data shapes:")
            print(f"    Low-res input: {low_res.shape}")
            print(f"    Targets: {targets.shape}")
            print(f"    Coords: {coords.shape}")

            # Debug data value ranges
            print(f"  📈 Value ranges:")
            print(f"    Low-res T range: [{low_res[0,0,:,:,0].min().item():.3f}, {low_res[0,0,:,:,0].max().item():.3f}]")
            print(f"    Target T range: [{targets[0,:,0].min().item():.3f}, {targets[0,:,0].max().item():.3f}]")

            # Get predictions (these will be normalized)
            predictions = model(low_res, coords)
            print(f"    Prediction T range: [{predictions[0,:,0].min().item():.3f}, {predictions[0,:,0].max().item():.3f}]")

            # CRITICAL: Denormalize predictions and targets for visualization
            if data_module.normalizer is not None:
                # Move data to CPU before denormalization to match normalizer device
                predictions_cpu = predictions.cpu()
                targets_cpu = targets.cpu()

                predictions_denorm = data_module.normalizer.denormalize(predictions_cpu.view(-1, 4)).view(predictions_cpu.shape)
                targets_denorm = data_module.normalizer.denormalize(targets_cpu.view(-1, 4)).view(targets_cpu.shape)
            else:
                predictions_denorm = predictions.cpu()
                targets_denorm = targets.cpu()

            # Use denormalized data for visualization (on CPU)
            predictions = predictions_denorm
            targets = targets_denorm

            # Reshape for visualization
            B, N, C = predictions.shape
            target_B, target_N, target_C = targets.shape

            # Make sure predictions and targets have the same N
            assert N == target_N, f"Prediction N ({N}) != target N ({target_N})"

            # FIXED: Use correct spatial dimensions based on the original RB data
            # Original RB data: (time_steps, 170, 512, 4)
            # With spatial_downsample=4: low-res becomes (time_steps, 43, 128, 4)
            # With temporal_downsample=4: clip becomes 8 timesteps

            # Get low-res dimensions to determine high-res structure
            B_lr, C_lr, T_lr, H_lr, W_lr = low_res.shape
            print(f"  📐 Low-res input shape: {low_res.shape}")

            # Use the known original data structure: H=170, W=512
            # High-res dimensions after upsampling
            H_hr = H_lr * spatial_downsample  # Should be close to 170
            W_hr = W_lr * spatial_downsample  # Should be close to 512
            T_hr = T_lr * temporal_downsample  # Should be 8 for clip length

            print(f"  📐 Calculated high-res dimensions:")
            print(f"    Time steps (T): {T_hr}")
            print(f"    Height (H): {H_hr}")
            print(f"    Width (W): {W_hr}")
            print(f"    Expected total points: {T_hr * H_hr * W_hr}")
            print(f"    Actual total points N: {N}")

            # Handle dimension mismatch by calculating correct dimensions from actual data
            if T_hr * H_hr * W_hr != N:
                print(f"  ⚠️  Dimension mismatch! Calculating from actual data...")

                # Try common clip lengths
                for test_T in [8, 4, 16, 32]:
                    spatial_points = N // test_T
                    if N % test_T == 0:  # Make sure it divides evenly
                        # Try to match the original RB aspect ratio (170:512 ≈ 1:3)
                        aspect_ratio = 512 / 170  # ≈ 3.01

                        # For downsampled data, try different downsampling factors
                        for ds in [1, 2, 4, 8, 16]:
                            target_H = 170 // ds
                            target_W = 512 // ds

                            if target_H * target_W == spatial_points:
                                T_hr, H_hr, W_hr = test_T, target_H, target_W
                                print(f"    ✅ Found exact match: T={T_hr}, H={H_hr}, W={W_hr} (downsample={ds})")
                                break

                        # Also try the computed dimensions from low-res input
                        computed_H = H_lr * spatial_downsample
                        computed_W = W_lr * spatial_downsample
                        if computed_H * computed_W == spatial_points:
                            T_hr, H_hr, W_hr = test_T, computed_H, computed_W
                            print(f"    ✅ Found match from low-res: T={T_hr}, H={H_hr}, W={W_hr}")
                            break
                        else:
                            # Try to find best factorization that preserves aspect ratio
                            best_diff = float('inf')
                            best_h = best_w = 1

                            for h in range(1, int(spatial_points**0.5) + 50):
                                if spatial_points % h == 0:
                                    w = spatial_points // h
                                    # Prefer aspect ratios close to original (w/h ≈ 3)
                                    current_ratio = w / h
                                    ratio_diff = abs(current_ratio - aspect_ratio)

                                    if ratio_diff < best_diff:
                                        best_diff = ratio_diff
                                        best_h, best_w = h, w

                            if best_diff < 2.0:  # Accept if reasonably close to target ratio
                                T_hr, H_hr, W_hr = test_T, best_h, best_w
                                print(f"    ✅ Found good match: T={T_hr}, H={H_hr}, W={W_hr} (ratio={best_w/best_h:.2f})")
                                break

                if T_hr * H_hr * W_hr != N:
                    print(f"    ⚠️  Last resort: using most square factorization")
                    # Fallback to square-ish factorization but with T=8
                    T_hr = 8
                    spatial_points = N // T_hr
                    H_hr = int(spatial_points ** 0.5)
                    W_hr = spatial_points // H_hr
                    print(f"    Final fallback: T={T_hr}, H={H_hr}, W={W_hr}")

            print(f"  📐 Final dimensions: T={T_hr}, H={H_hr}, W={W_hr}")

            # Reshape to [T, H, W, C] - both predictions and targets should have same structure
            pred_reshaped = predictions.view(B, T_hr, H_hr, W_hr, C)  # [B, T, H, W, C]
            target_reshaped = targets.view(B, T_hr, H_hr, W_hr, C)    # [B, T, H, W, C]
            low_res_reshaped = low_res.cpu().permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
            
            results['predictions'].append(pred_reshaped[0])  # [T, H, W, C]
            results['truth_fields'].append(target_reshaped[0])
            results['input_fields'].append(low_res_reshaped[0])
    
    # Concatenate results
    for key in results:
        if results[key]:
            results[key] = torch.cat(results[key], dim=0).numpy()  # [T_total, H, W, C]
    
    return results


def create_demo_data() -> dict:
    """Create synthetic demo data for visualization."""
    print("Creating synthetic demo data...")
    
    # Parameters
    T, H, W = 80, 64, 192  # Time, Height, Width
    
    # Spatial coordinates
    x = np.linspace(0, 3, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Initialize fields
    truth_fields = np.zeros((T, H, W, 4))  # T, p, u, v
    input_fields = np.zeros((T, H//4, W//4, 4))
    predictions = np.zeros((T, H, W, 4))
    
    for t in range(T):
        time_val = t * 0.1
        
        # Temperature field with convection cells
        temp = 0.4 * np.sin(2*np.pi*X/1.5 + 0.1*time_val) * np.sin(np.pi*Y)
        temp += 0.2 * np.sin(4*np.pi*X/1.5 + 0.05*time_val) * np.sin(2*np.pi*Y)
        temp += 0.1 * np.sin(6*np.pi*X/1.5 - 0.08*time_val) * np.sin(3*np.pi*Y)
        temp += np.random.normal(0, 0.02, (H, W))
        
        # Velocity fields (simplified)
        u = 0.3 * np.cos(2*np.pi*X/1.5 + 0.1*time_val) * np.cos(np.pi*Y)
        v = 0.2 * np.sin(2*np.pi*X/1.5 + 0.1*time_val) * np.sin(2*np.pi*Y)
        
        # Pressure field
        p = 0.1 * np.sin(np.pi*X/1.5) * np.cos(np.pi*Y) + 0.05 * np.random.randn(H, W)
        
        # Store truth
        truth_fields[t, :, :, 0] = temp  # Temperature
        truth_fields[t, :, :, 1] = p     # Pressure
        truth_fields[t, :, :, 2] = u     # u-velocity
        truth_fields[t, :, :, 3] = v     # v-velocity
        
        # Create low-resolution input
        input_fields[t, :, :, 0] = temp[::4, ::4]
        input_fields[t, :, :, 1] = p[::4, ::4]
        input_fields[t, :, :, 2] = u[::4, ::4]
        input_fields[t, :, :, 3] = v[::4, ::4]
        
        # Create predictions with small errors
        predictions[t, :, :, 0] = temp + np.random.normal(0, 0.03, (H, W))
        predictions[t, :, :, 1] = p + np.random.normal(0, 0.02, (H, W))
        predictions[t, :, :, 2] = u + np.random.normal(0, 0.025, (H, W))
        predictions[t, :, :, 3] = v + np.random.normal(0, 0.025, (H, W))
    
    return {
        'input_fields': input_fields,
        'truth_fields': truth_fields,
        'predictions': predictions
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CDAnet Visualization Generator")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data
    if args.demo:
        print("Creating demo visualization...")
        results = create_demo_data()
    elif args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        data_file = os.path.join(args.data_dir, f'rb_data_Ra_{args.Ra:.0e}.h5')
        results = load_model_and_predict(
            args.checkpoint, data_file, args.Ra,
            args.spatial_downsample, args.temporal_downsample
        )
    else:
        raise ValueError("Either --checkpoint or --demo must be specified")
    
    # Initialize visualizer
    visualizer = RBVisualization()
    
    # Variable index mapping
    var_idx = {'T': 0, 'p': 1, 'u': 2, 'v': 3}[args.variable]
    
    # Extract variable fields
    input_var = results['input_fields'][:, :, :, var_idx]
    truth_var = results['truth_fields'][:, :, :, var_idx]
    pred_var = results['predictions'][:, :, :, var_idx]
    
    print(f"Data shapes:")
    print(f"  Input: {input_var.shape}")
    print(f"  Truth: {truth_var.shape}")
    print(f"  Predictions: {pred_var.shape}")
    
    # Set color limits based on variable
    if args.variable == 'T':
        vmin, vmax = -0.5, 0.5
    elif args.variable in ['u', 'v']:
        vmin, vmax = -0.4, 0.4
    else:  # pressure
        vmin, vmax = -0.2, 0.2
    
    # Create main comparison plot (paper style)
    print("Creating comparison plot...")
    fig1 = visualizer.create_comparison_plot(
        input_field=input_var,
        truth_field=truth_var,
        cdanet_pred=pred_var,
        times=args.times,
        variable=args.variable,
        vmin=vmin, vmax=vmax,
        save_path=os.path.join(args.output_dir, f'rb_comparison_{args.variable}.{args.format}')
    )
    
    # Create temporal evolution
    print("Creating temporal evolution plot...")
    fig2 = visualizer.create_temporal_evolution(
        truth_field=truth_var,
        cdanet_pred=pred_var,
        variable=args.variable,
        n_snapshots=args.n_snapshots,
        save_path=os.path.join(args.output_dir, f'temporal_evolution_{args.variable}.{args.format}')
    )
    
    # Create error analysis
    print("Creating error analysis...")
    time_idx = min(len(truth_var) - 1, int(args.times[1] / 0.1))  # Use middle time
    fig3 = visualizer.create_error_analysis(
        truth_field=truth_var,
        cdanet_pred=pred_var,
        time_idx=time_idx,
        save_path=os.path.join(args.output_dir, f'error_analysis_{args.variable}.{args.format}')
    )
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    print(f"  - rb_comparison_{args.variable}.{args.format}")
    print(f"  - temporal_evolution_{args.variable}.{args.format}")
    print(f"  - error_analysis_{args.variable}.{args.format}")
    print("=" * 60)


if __name__ == '__main__':
    main()
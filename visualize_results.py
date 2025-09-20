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
    print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")

    # Create model
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        model_config = config_dict['model']
        print(f"🔍 Using config from checkpoint: {model_config}")
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
        print(f"🔍 Using default config: {model_config}")

    model = CDAnet(**model_config)

    # Check if state dict loads correctly
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model state dict loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model state dict: {e}")
        print(f"Model state dict keys: {list(model.state_dict().keys())[:5]}...")
        if 'model_state_dict' in checkpoint:
            print(f"Checkpoint state dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")

    model.eval()

    # Check if model parameters look reasonable
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔍 Total model parameters: {total_params:,}")

    # Check a few parameter values
    for name, param in model.named_parameters():
        print(f"🔍 {name}: shape {param.shape}, mean {param.mean().item():.6f}, std {param.std().item():.6f}")
        if len(list(model.named_parameters())) > 5:  # Only show first few
            break
    
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
            print(f"  📈 Value ranges (BEFORE denormalization):")
            print(f"    Low-res T range: [{low_res[0,0,:,:,0].min().item():.3f}, {low_res[0,0,:,:,0].max().item():.3f}]")
            print(f"    Target T range: [{targets[0,:,0].min().item():.3f}, {targets[0,:,0].max().item():.3f}]")
            print(f"    Low-res T mean/std: {low_res[0,0,:,:,0].mean().item():.3f}/{low_res[0,0,:,:,0].std().item():.3f}")
            print(f"    Target T mean/std: {targets[0,:,0].mean().item():.3f}/{targets[0,:,0].std().item():.3f}")

            # 🔍 DEBUG: Check target data structure before reshaping
            print(f"  🔍 Target data analysis:")
            print(f"    Target tensor shape: {targets.shape}")  # Should be [1, N, 4]
            target_T_flat = targets[0, :, 0]  # [N] - flattened temperature field
            print(f"    Flat target T: min={target_T_flat.min():.3f}, max={target_T_flat.max():.3f}")

            # Check if the flattened data shows any patterns
            print(f"    First 10 values: {target_T_flat[:10].tolist()}")
            print(f"    Values 1000-1010: {target_T_flat[1000:1010].tolist()}")

            # Check variance in chunks
            chunk_size = 1000
            chunk_vars = []
            for i in range(0, min(10000, len(target_T_flat)), chunk_size):
                chunk = target_T_flat[i:i+chunk_size]
                chunk_vars.append(chunk.var().item())
            print(f"    Variance in chunks: {chunk_vars[:5]}")  # Show first 5 chunks

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

            # Debug denormalized data ranges
            print(f"  📈 Value ranges (AFTER denormalization):")
            print(f"    Target T range: [{targets[0,:,0].min().item():.3f}, {targets[0,:,0].max().item():.3f}]")
            print(f"    Target T mean/std: {targets[0,:,0].mean().item():.3f}/{targets[0,:,0].std().item():.3f}")
            print(f"    Prediction T range: [{predictions[0,:,0].min().item():.3f}, {predictions[0,:,0].max().item():.3f}]")
            print(f"    Prediction T mean/std: {predictions[0,:,0].mean().item():.3f}/{predictions[0,:,0].std().item():.3f}")

            # Check if denormalization worked correctly
            if data_module.normalizer is not None:
                norm_stats = data_module.normalizer
                print(f"  📊 Normalizer stats:")
                print(f"    Mean: {norm_stats.mean}")
                print(f"    Std: {norm_stats.std}")
            else:
                print(f"  ⚠️  No normalizer found!")

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

            # 🔧 CRITICAL FIX: The data layout might be different than assumed
            # Try different reshape orders to find the correct one

            print(f"  🔧 Trying different reshape orders...")

            # Original attempt
            target_reshaped_v1 = targets.view(B, T_hr, H_hr, W_hr, C)
            first_frame_v1 = target_reshaped_v1[0, 0, :, :, 0]
            h_var_v1 = first_frame_v1.var(dim=1).mean().item()
            v_var_v1 = first_frame_v1.var(dim=0).mean().item()
            print(f"    V1 (T,H,W): h_var={h_var_v1:.6f}, v_var={v_var_v1:.6f}")

            # Try H,W,T order instead
            try:
                target_reshaped_v2 = targets.view(B, H_hr, W_hr, T_hr, C).permute(0, 3, 1, 2, 4)  # [B,T,H,W,C]
                first_frame_v2 = target_reshaped_v2[0, 0, :, :, 0]
                h_var_v2 = first_frame_v2.var(dim=1).mean().item()
                v_var_v2 = first_frame_v2.var(dim=0).mean().item()
                print(f"    V2 (H,W,T->T,H,W): h_var={h_var_v2:.6f}, v_var={v_var_v2:.6f}")
            except:
                target_reshaped_v2 = target_reshaped_v1
                h_var_v2, v_var_v2 = h_var_v1, v_var_v1

            # Try W,H,T order
            try:
                target_reshaped_v3 = targets.view(B, W_hr, H_hr, T_hr, C).permute(0, 3, 2, 1, 4)  # [B,T,H,W,C]
                first_frame_v3 = target_reshaped_v3[0, 0, :, :, 0]
                h_var_v3 = first_frame_v3.var(dim=1).mean().item()
                v_var_v3 = first_frame_v3.var(dim=0).mean().item()
                print(f"    V3 (W,H,T->T,H,W): h_var={h_var_v3:.6f}, v_var={v_var_v3:.6f}")
            except:
                target_reshaped_v3 = target_reshaped_v1
                h_var_v3, v_var_v3 = h_var_v1, v_var_v1

            # Choose the version with most balanced variance (indicates proper 2D structure)
            variance_ratios = [
                (abs(h_var_v1 - v_var_v1), target_reshaped_v1, "T,H,W"),
                (abs(h_var_v2 - v_var_v2), target_reshaped_v2, "H,W,T->T,H,W"),
                (abs(h_var_v3 - v_var_v3), target_reshaped_v3, "W,H,T->T,H,W")
            ]

            # Also prefer higher overall variance (more structure)
            overall_vars = [
                (h_var_v1 + v_var_v1, target_reshaped_v1, "T,H,W"),
                (h_var_v2 + v_var_v2, target_reshaped_v2, "H,W,T->T,H,W"),
                (h_var_v3 + v_var_v3, target_reshaped_v3, "W,H,T->T,H,W")
            ]

            # Choose version with highest overall variance (most structure)
            best_overall_var, target_reshaped, best_order = max(overall_vars, key=lambda x: x[0])
            print(f"    ✅ Selected: {best_order} (overall_var={best_overall_var:.6f})")

            # Apply same reshaping to predictions
            if best_order == "H,W,T->T,H,W":
                pred_reshaped = predictions.view(B, H_hr, W_hr, T_hr, C).permute(0, 3, 1, 2, 4)
            elif best_order == "W,H,T->T,H,W":
                pred_reshaped = predictions.view(B, W_hr, H_hr, T_hr, C).permute(0, 3, 2, 1, 4)
            else:
                pred_reshaped = predictions.view(B, T_hr, H_hr, W_hr, C)

            # 🔧 FIXED: Also denormalize low-res input for fair comparison
            low_res_cpu = low_res.cpu().permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]

            # Denormalize low-res input if normalizer exists
            if data_module.normalizer is not None:
                B_lr, T_lr, H_lr, W_lr, C_lr = low_res_cpu.shape
                low_res_flat = low_res_cpu.view(-1, C_lr)
                low_res_denorm = data_module.normalizer.denormalize(low_res_flat).view(low_res_cpu.shape)
                low_res_reshaped = low_res_denorm
                print(f"  🔧 Denormalized low-res input: range [{low_res_reshaped[0,:,:,:,0].min():.3f}, {low_res_reshaped[0,:,:,:,0].max():.3f}]")
            else:
                low_res_reshaped = low_res_cpu
                print(f"  ⚠️  No normalizer - using raw low-res")
            
            # 🔍 DEBUG: Check reshaped truth data quality
            truth_sample = target_reshaped[0]  # [T, H, W, C]
            print(f"  🔍 Truth after reshape: shape {truth_sample.shape}")
            print(f"    T field stats: min={truth_sample[:,:,:,0].min():.3f}, max={truth_sample[:,:,:,0].max():.3f}")
            timestep_means = [truth_sample[t,:,:,0].mean().item() for t in range(min(3, truth_sample.shape[0]))]
            print(f"    T field mean per timestep: {[f'{x:.3f}' for x in timestep_means]}")

            # Check if truth has the stripe problem
            first_timestep = truth_sample[0, :, :, 0]  # [H, W]
            print(f"    First timestep shape: {first_timestep.shape}")

            # Check for horizontal stripes (values should vary across width)
            horizontal_variance = first_timestep.var(dim=1).mean()  # Variance across width for each row
            vertical_variance = first_timestep.var(dim=0).mean()    # Variance across height for each column
            print(f"    Horizontal variance: {horizontal_variance:.6f}")
            print(f"    Vertical variance: {vertical_variance:.6f}")

            if horizontal_variance < 0.001:
                print(f"  ⚠️  WARNING: Truth shows horizontal stripe pattern!")

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
    
    print(f"Final visualization data shapes:")
    print(f"  Input: {input_var.shape}")
    print(f"  Truth: {truth_var.shape}")
    print(f"  Predictions: {pred_var.shape}")

    print(f"Final visualization data ranges:")
    print(f"  Input T range: [{input_var.min():.3f}, {input_var.max():.3f}] mean: {input_var.mean():.3f}")
    print(f"  Truth T range: [{truth_var.min():.3f}, {truth_var.max():.3f}] mean: {truth_var.mean():.3f}")
    print(f"  Pred T range: [{pred_var.min():.3f}, {pred_var.max():.3f}] mean: {pred_var.mean():.3f}")

    # Check for potential issues
    if abs(truth_var.mean()) < 0.1 and truth_var.std() < 0.1:
        print(f"  ⚠️  WARNING: Truth data seems to have very small values!")
        print(f"  Truth std: {truth_var.std():.3f}")

    if input_var.max() - input_var.min() > truth_var.max() - truth_var.min():
        print(f"  ⚠️  WARNING: Input has larger range than Truth - this is unusual!")
    
    # Set color limits based on actual data range
    all_data = np.concatenate([truth_var.flatten(), pred_var.flatten()])
    data_min, data_max = all_data.min(), all_data.max()
    data_range = data_max - data_min

    print(f"Actual data range: [{data_min:.3f}, {data_max:.3f}], span: {data_range:.3f}")

    # 🎨 FIXED: Use separate optimized color ranges for better visualization
    if args.variable == 'T':
        input_min, input_max = input_var.min(), input_var.max()
        truth_min, truth_max = truth_var.min(), truth_var.max()
        pred_min, pred_max = pred_var.min(), pred_var.max()

        print(f"Data ranges before color mapping:")
        print(f"  Input: [{input_min:.3f}, {input_max:.3f}] span={input_max-input_min:.3f}")
        print(f"  Truth: [{truth_min:.3f}, {truth_max:.3f}] span={truth_max-truth_min:.3f}")
        print(f"  Pred:  [{pred_min:.3f}, {pred_max:.3f}] span={pred_max-pred_min:.3f}")

        # 🔧 SOLUTION: Use data-aware color mapping for both Input and Truth

        # Calculate percentiles for both Input and Truth
        input_flat = input_var.flatten()
        truth_flat = truth_var.flatten()

        input_p5, input_p95 = np.percentile(input_flat, [5, 95])
        truth_p5, truth_p95 = np.percentile(truth_flat, [5, 95])

        input_median = np.median(input_flat)
        truth_median = np.median(truth_flat)

        print(f"🎨 Data-aware color mapping:")
        print(f"  Input p5-p95: [{input_p5:.3f}, {input_p95:.3f}], median: {input_median:.3f}")
        print(f"  Truth p5-p95: [{truth_p5:.3f}, {truth_p95:.3f}], median: {truth_median:.3f}")

        # Strategy: Use a color range that shows both datasets well
        # Take the wider range but ensure both datasets are visible

        # Find the range that includes both datasets' main content
        combined_p5 = min(input_p5, truth_p5)
        combined_p95 = max(input_p95, truth_p95)
        combined_median = (input_median + truth_median) / 2

        # Create symmetric range around combined median
        range_span = max(combined_p95 - combined_median, combined_median - combined_p5)

        vmin = combined_median - range_span * 1.1  # Small padding
        vmax = combined_median + range_span * 1.1

        print(f"  Combined range: [{vmin:.3f}, {vmax:.3f}], center: {combined_median:.3f}")

        # Safety check: ensure range is reasonable
        if vmax - vmin < 0.1:  # Too narrow
            print(f"  ⚠️  Range too narrow, using wider default")
            vmin, vmax = combined_median - 0.3, combined_median + 0.3

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
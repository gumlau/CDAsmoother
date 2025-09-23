#!/usr/bin/env python3
"""
Quick fix script to convert your existing RB data to reference format.
This will fix channel ordering and create consolidated dataset for training.
"""

import h5py
import numpy as np
import os
import glob

def fix_data_format(input_dir='rb_data_numerical', output_file=None):
    """Convert existing data to reference format [p,b,u,w]."""

    # Find data files
    pattern = os.path.join(input_dir, 'rb_data_Ra_*.h5')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No data files found in {input_dir}")
        return False

    print(f"Found {len(files)} data files")

    # Load first file to get dimensions
    with h5py.File(files[0], 'r') as f:
        print(f"File structure: {list(f.keys())}")
        print(f"Attributes: {dict(f.attrs)}")

        if 'data' in f:
            data_shape = f['data'].shape
            print(f"Data shape: {data_shape}")
            # Expecting: [n_samples, ny, nx, 4] where 4 channels are [T, p, u, v]
            n_samples, ny, nx, n_channels = data_shape
        else:
            print("No 'data' key found in file")
            return False

    print(f"Data dimensions: {n_samples} samples × {ny}×{nx} grid × {n_channels} channels")

    # Process all files
    all_p_data = []
    all_b_data = []  # temperature -> b (buoyancy)
    all_u_data = []  # velocity_x -> u
    all_w_data = []  # velocity_y -> w

    for file_idx, filepath in enumerate(files):
        print(f"Processing file {file_idx+1}/{len(files)}: {os.path.basename(filepath)}")

        with h5py.File(filepath, 'r') as f:
            # Your data format: [n_samples, ny, nx, 4] where channels are [T, p, u, v]
            data = f['data'][:]  # Shape: [n_samples, ny, nx, 4]

            # Extract channels: your format is [T, p, u, v]
            # Convert to reference format [p, b, u, w]
            T_data = data[:, :, :, 0]  # Temperature -> b (buoyancy)
            p_data = data[:, :, :, 1]  # Pressure -> p
            u_data = data[:, :, :, 2]  # U-velocity -> u
            v_data = data[:, :, :, 3]  # V-velocity -> w

            # Add to collections
            all_p_data.append(p_data)
            all_b_data.append(T_data)  # Temperature becomes 'b' (buoyancy)
            all_u_data.append(u_data)
            all_w_data.append(v_data)  # V-velocity becomes 'w'

    # Stack all files: [n_files, n_samples, ny, nx]
    p_data = np.stack(all_p_data, axis=0)
    b_data = np.stack(all_b_data, axis=0)
    u_data = np.stack(all_u_data, axis=0)
    w_data = np.stack(all_w_data, axis=0)

    print(f"Final data shapes: p={p_data.shape}, b={b_data.shape}, u={u_data.shape}, w={w_data.shape}")

    # Check data ranges
    print(f"Data ranges:")
    print(f"  Pressure: [{p_data.min():.3f}, {p_data.max():.3f}]")
    print(f"  Temperature: [{b_data.min():.3f}, {b_data.max():.3f}]")
    print(f"  U-velocity: [{u_data.min():.3f}, {u_data.max():.3f}]")
    print(f"  V-velocity: [{w_data.min():.3f}, {w_data.max():.3f}]")

    # Replace NaN/inf with zeros if present
    for name, data in [('p', p_data), ('b', b_data), ('u', u_data), ('w', w_data)]:
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"Warning: Fixing invalid values in {name}")
            data[np.isnan(data) | np.isinf(data)] = 0.0

    # Save in consolidated format
    if output_file is None:
        output_file = os.path.join(input_dir, 'rb2d_ra1e+05_consolidated.h5')

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('p', data=p_data, compression='gzip')
        f.create_dataset('b', data=b_data, compression='gzip')
        f.create_dataset('u', data=u_data, compression='gzip')
        f.create_dataset('w', data=w_data, compression='gzip')

        # Metadata matching reference
        f.attrs['Ra'] = 1e5
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_files'] = len(files)
        f.attrs['n_samples'] = n_samples
        f.attrs['format'] = 'reference_compatible'

    print(f"✅ Fixed dataset saved: {output_file}")
    print(f"   Format: [p,b,u,w] × {len(files)} files × {n_samples} samples × {ny}×{nx}")

    return True

def create_training_config():
    """Create a working training configuration."""
    config = """
# Quick training test command:
python3 train_cdanet_fixed.py \\
    --data_folder ./rb_data_numerical \\
    --train_data rb2d_ra1e+05_consolidated.h5 \\
    --eval_data rb2d_ra1e+05_consolidated.h5 \\
    --epochs 20 \\
    --batch_size 2 \\
    --nx 128 --nz 64 --nt 16 \\
    --downsamp_xz 4 --downsamp_t 4 \\
    --lr 0.01 \\
    --alpha_pde 0.01 \\
    --output_folder ./outputs_fixed

# For full training:
python3 train_cdanet_fixed.py \\
    --epochs 100 \\
    --batch_size 4 \\
    --lr 0.1 \\
    --output_folder ./outputs_full
"""

    with open('training_commands.sh', 'w') as f:
        f.write(config)

    print("✅ Training commands saved to: training_commands.sh")

if __name__ == '__main__':
    print("🔧 Fixing Existing RB Data Format")
    print("=" * 50)

    # Fix data format
    if fix_data_format():
        print("\n✅ Data format fixed successfully!")

        # Create training config
        create_training_config()

        print("\n🚀 Next steps:")
        print("1. Run: bash training_commands.sh")
        print("2. Monitor training in ./outputs_fixed/tensorboard/")
        print("3. Check results with visualize_results.py")

    else:
        print("❌ Failed to fix data format")
#!/usr/bin/env python3
"""
Convert multiple RB run files to consolidated format for CDAnet training
"""

import numpy as np
import h5py
import os
import glob

def convert_rb_data_to_cdanet_format(data_dir='rb_data_numerical', Ra=1e5):
    """Convert multiple RB run files to single consolidated file - optimized version"""
    
    print(f"Converting RB data for Ra = {Ra:.0e}")
    
    # Find all run files for this Ra
    pattern = os.path.join(data_dir, f"rb_data_Ra_{Ra:.0e}_run_*.h5")
    run_files = sorted(glob.glob(pattern))
    
    if not run_files:
        print(f"No run files found for Ra = {Ra:.0e}")
        return None
    
    print(f"Found {len(run_files)} run files")
    
    # Scan all files to get accurate sample count and metadata
    total_samples = 0
    metadata = {}
    file_info = []
    
    for i, filepath in enumerate(run_files):
        with h5py.File(filepath, 'r') as f:
            n_samples = f.attrs['n_samples']
            total_samples += n_samples
            
            # Get metadata from first file
            if i == 0:
                metadata = {
                    'Ra': f.attrs['Ra'],
                    'nx': f.attrs['nx'], 
                    'ny': f.attrs['ny'],
                    'Lx': f.attrs['Lx'],
                    'Ly': f.attrs['Ly']
                }
                
                # Determine data format and shape
                if 'data' in f:
                    sample_shape = f['data'].shape[1:]  # (ny, nx, 4)
                    is_optimized = True
                    print(f"  Detected optimized format with shape: {sample_shape}")
                else:
                    # Old format with individual frame groups
                    frame_grp = f['frame_000']
                    T_shape = frame_grp['temperature'].shape
                    sample_shape = (*T_shape, 4)
                    is_optimized = False
            
            file_info.append({
                'path': filepath,
                'n_samples': n_samples,
                'optimized': 'data' in f
            })
    
    print(f"Total samples across all files: {total_samples}")
    
    # Pre-allocate output array
    print(f"Pre-allocating array for {total_samples} samples...")
    consolidated_data = np.zeros((total_samples,) + sample_shape, dtype=np.float32)
    
    # Load data efficiently
    sample_idx = 0
    for i, file_data in enumerate(file_info):
        filepath = file_data['path']
        n_frames = file_data['n_samples']
        is_file_optimized = file_data['optimized']
        
        print(f"  Loading run {i+1}/{len(file_info)}: {os.path.basename(filepath)}")
        
        with h5py.File(filepath, 'r') as f:
            end_idx = sample_idx + n_frames
            
            if is_file_optimized:
                # New optimized format - direct copy with shape validation
                file_data_shape = f['data'].shape
                expected_shape = (n_frames,) + sample_shape

                if file_data_shape != expected_shape:
                    print(f"    Warning: Shape mismatch! File: {file_data_shape}, Expected: {expected_shape}")
                    # 如果形状不匹配，需要重新构建sample_shape
                    sample_shape = file_data_shape[1:]
                    print(f"    Updating sample_shape to: {sample_shape}")

                    # 重新创建consolidated_data数组
                    if i == 0:  # 只在第一个文件时重新创建
                        print(f"    Recreating consolidated_data with corrected shape")
                        del consolidated_data  # 删除旧数组
                        consolidated_data = np.zeros((total_samples,) + sample_shape, dtype=np.float32)

                consolidated_data[sample_idx:end_idx] = f['data'][:]
            else:
                # Old format - need to reconstruct
                for frame_idx in range(n_frames):
                    frame_grp = f[f'frame_{frame_idx:03d}']
                    
                    # Stack fields: [T, p, u, v] -> shape (H, W, 4)
                    T = frame_grp['temperature'][:]
                    p = frame_grp['pressure'][:]
                    u = frame_grp['velocity_x'][:]
                    v = frame_grp['velocity_y'][:]
                    
                    consolidated_data[sample_idx + frame_idx] = np.stack([T, p, u, v], axis=-1)
            
            sample_idx += n_frames
    
    print(f"Consolidated data shape: {consolidated_data.shape}")
    
    # Save consolidated file with optimized compression
    output_file = os.path.join(data_dir, f"rb_data_Ra_{Ra:.0e}.h5")
    
    with h5py.File(output_file, 'w') as f:
        # Save data with chunking for better performance
        chunk_size = min(100, consolidated_data.shape[0])
        f.create_dataset('data', data=consolidated_data, 
                        compression='gzip', compression_opts=6,
                        chunks=(chunk_size, *sample_shape))
        
        # Save metadata
        for key, value in metadata.items():
            f.attrs[key] = value
        
        f.attrs['total_frames'] = total_samples
        f.attrs['n_runs'] = len(run_files)
        f.attrs['optimized_format'] = True
        
        print(f"✅ Saved consolidated data to {output_file}")
        print(f"   Total frames: {total_samples}")
        print(f"   Data shape: {consolidated_data.shape}")
        print(f"   Memory size: {consolidated_data.nbytes / (1024**3):.2f} GB")
    
    return output_file

def main():
    """Convert all available RB data"""
    data_dir = 'rb_data_numerical'
    
    # Find all Ra numbers with data
    pattern = os.path.join(data_dir, "rb_data_Ra_*_run_*.h5")
    all_files = glob.glob(pattern)
    
    # Extract unique Ra numbers
    Ra_numbers = set()
    for filepath in all_files:
        basename = os.path.basename(filepath)
        # Extract Ra from filename like "rb_data_Ra_1e+05_run_01.h5"
        parts = basename.split('_')
        for i, part in enumerate(parts):
            if part == 'Ra' and i+1 < len(parts):
                Ra_str = parts[i+1]
                try:
                    Ra = float(Ra_str.replace('+', ''))
                    Ra_numbers.add(Ra)
                    break
                except ValueError:
                    continue
    
    print(f"Found data for Ra numbers: {sorted(Ra_numbers)}")
    
    # Convert each Ra
    for Ra in sorted(Ra_numbers):
        convert_rb_data_to_cdanet_format(data_dir, Ra)
        print()

if __name__ == "__main__":
    main()
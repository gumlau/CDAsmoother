#!/usr/bin/env python3
"""
Convert multiple RB run files to consolidated format for CDAnet training
"""

import numpy as np
import h5py
import os
import glob

def convert_rb_data_to_cdanet_format(data_dir='rb_data_numerical', Ra=1e5):
    """Convert multiple RB run files to single consolidated file"""
    
    print(f"Converting RB data for Ra = {Ra:.0e}")
    
    # Find all run files for this Ra
    pattern = os.path.join(data_dir, f"rb_data_Ra_{Ra:.0e}_run_*.h5")
    run_files = sorted(glob.glob(pattern))
    
    if not run_files:
        print(f"No run files found for Ra = {Ra:.0e}")
        return None
    
    print(f"Found {len(run_files)} run files")
    
    # Load all data
    all_data = []
    metadata = {}
    
    for i, filepath in enumerate(run_files):
        print(f"  Loading run {i+1}/{len(run_files)}: {os.path.basename(filepath)}")
        
        with h5py.File(filepath, 'r') as f:
            # Get metadata from first file
            if i == 0:
                metadata['Ra'] = f.attrs['Ra']
                metadata['nx'] = f.attrs['nx']
                metadata['ny'] = f.attrs['ny']
                metadata['Lx'] = f.attrs['Lx']
                metadata['Ly'] = f.attrs['Ly']
                metadata['n_samples'] = f.attrs['n_samples']
                
            # Load all frames from this run
            run_data = []
            n_frames = f.attrs['n_samples']
            
            for frame_idx in range(n_frames):
                frame_grp = f[f'frame_{frame_idx:03d}']
                
                # Stack fields: [T, p, u, v] -> shape (H, W, 4)
                T = frame_grp['temperature'][:]
                p = frame_grp['pressure'][:]
                u = frame_grp['velocity_u'][:]
                v = frame_grp['velocity_v'][:]
                
                # Stack along last dimension: (H, W, 4)
                frame_data = np.stack([T, p, u, v], axis=-1)
                run_data.append(frame_data)
            
            all_data.extend(run_data)
    
    # Convert to numpy array: [T, H, W, 4]
    consolidated_data = np.array(all_data)
    print(f"Consolidated data shape: {consolidated_data.shape}")
    
    # Save consolidated file
    output_file = os.path.join(data_dir, f"rb_data_Ra_{Ra:.0e}.h5")
    
    with h5py.File(output_file, 'w') as f:
        # Save data
        f.create_dataset('data', data=consolidated_data, compression='gzip')
        
        # Save metadata
        for key, value in metadata.items():
            f.attrs[key] = value
        
        f.attrs['total_frames'] = len(all_data)
        f.attrs['n_runs'] = len(run_files)
        
        print(f"✅ Saved consolidated data to {output_file}")
        print(f"   Total frames: {len(all_data)}")
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
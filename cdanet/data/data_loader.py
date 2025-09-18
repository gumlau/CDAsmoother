"""
Data loading utilities and preprocessing for CDAnet training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional
from .dataset import RBDataset, RandomCoordinateSampler, DataNormalizer


class ChainedTransform:
    """Picklable transform composition for two transforms."""
    def __init__(self, first, second):
        self.first = first
        self.second = second
        
    def __call__(self, sample):
        sample = self.first(sample)
        sample = self.second(sample)
        return sample


class ComposedTransform:
    """Picklable transform composition for multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RBDataModule:
    """
    Data module for Rayleigh-Bénard convection datasets.
    Handles data loading, preprocessing, and splitting.
    
    Args:
        data_dir: Directory containing HDF5 data files
        spatial_downsample: Spatial downsampling factor
        temporal_downsample: Temporal downsampling factor
        clip_length: Length of temporal clips
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pde_points: Number of points for PDE loss computation
        normalize: Whether to normalize the data
    """
    
    def __init__(self, data_dir: str, spatial_downsample: int = 4, temporal_downsample: int = 4,
                 clip_length: int = 8, batch_size: int = 32, num_workers: int = 4,
                 pde_points: int = 3000, normalize: bool = True):  # 3000 points as per paper
        
        self.data_dir = data_dir
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pde_points = pde_points
        self.normalize = normalize
        
        self.normalizer = None
        self.datasets = {}
        self.data_loaders = {}
        
    def setup(self, Ra_numbers: List[float] = [1e5]):
        """
        Setup datasets for different Rayleigh numbers.
        
        Args:
            Ra_numbers: List of Rayleigh numbers to load
        """
        print(f"Setting up data module for Ra numbers: {Ra_numbers}")
        
        for Ra in Ra_numbers:
            # Find data file for this Ra number
            data_file = self._find_data_file(Ra)

            if data_file is None:
                print(f"Error: No data file found for Ra = {Ra}")
                print(f"  Checked directory: {self.data_dir}")
                print(f"  Looking for: rb_data_Ra_{Ra:.0e}.h5")
                import os
                if os.path.exists(self.data_dir):
                    files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
                    print(f"  Available files: {files[:5]}...")  # Show first 5 files
                continue

            import os
            print(f"Loading data for Ra = {Ra} from {data_file}")
            print(f"File exists: {os.path.exists(data_file)}")
            print(f"File size: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")
            
            # Create transforms
            transforms = [RandomCoordinateSampler(self.pde_points)]
            
            # Setup datasets for train/val/test
            for split in ['train', 'val', 'test']:
                try:
                    dataset = RBDataset(
                        data_path=data_file,
                        spatial_downsample=self.spatial_downsample,
                        temporal_downsample=self.temporal_downsample,
                        clip_length=self.clip_length,
                        split=split,
                        transform=self._compose_transforms(transforms.copy())
                    )

                    key = f"Ra_{Ra:.0e}_{split}"
                    self.datasets[key] = dataset
                    print(f"  Created dataset {key}: {len(dataset)} samples")

                except Exception as e:
                    print(f"  Error creating dataset for {split}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
            # Setup normalizer using training data
            if self.normalize and f"Ra_{Ra:.0e}_train" in self.datasets:
                self._setup_normalizer(self.datasets[f"Ra_{Ra:.0e}_train"])
                
        self._create_data_loaders()
        
    def _find_data_file(self, Ra: float) -> Optional[str]:
        """Find HDF5 data file(s) for given Rayleigh number."""
        import os
        import glob

        print(f"  Searching for Ra={Ra} in directory: {self.data_dir}")

        # First try single files
        possible_names = [
            f"rb_data_Ra_{Ra:.0e}.h5",
            f"rb_data_Ra_{int(Ra)}.h5",
            f"rb_data_{Ra:.0e}.h5",
            f"rb_data_{int(Ra)}.h5",
            f"Ra_{Ra:.0e}.h5",
            f"Ra_{int(Ra)}.h5",
            "rb_simulation_data.h5"  # Default name
        ]

        for name in possible_names:
            filepath = os.path.join(self.data_dir, name)
            print(f"    Checking: {filepath} -> {os.path.exists(filepath)}")
            if os.path.exists(filepath):
                print(f"    Found: {filepath}")
                return filepath

        # Look for multiple run files pattern
        pattern = os.path.join(self.data_dir, f"rb_data_Ra_{Ra:.0e}_run_*.h5")
        run_files = glob.glob(pattern)
        print(f"    Checking run files pattern: {pattern} -> found {len(run_files)} files")

        if run_files:
            # Return first file found - dataset will handle multiple runs
            print(f"    Using first run file: {run_files[0]}")
            return run_files[0]

        # Check if directory exists and list all files for debugging
        if os.path.exists(self.data_dir):
            all_files = os.listdir(self.data_dir)
            h5_files = [f for f in all_files if f.endswith('.h5')]
            print(f"    Available .h5 files in directory: {h5_files}")

            # Try to find any file with the Ra number in the name
            for filename in h5_files:
                if f"{Ra:.0e}" in filename or f"{int(Ra)}" in filename:
                    filepath = os.path.join(self.data_dir, filename)
                    print(f"    Found matching file by Ra number: {filepath}")
                    return filepath
        else:
            print(f"    Data directory does not exist: {self.data_dir}")

        print(f"    No data files found for Ra={Ra}")
        # If no files found, return None
        return None
        
    def _compose_transforms(self, transforms: List):
        """Compose multiple transforms."""
        return ComposedTransform(transforms)
    
    def _create_chained_transform(self, first_transform, second_transform):
        """Create a chained transform that can be pickled."""
        return ChainedTransform(first_transform, second_transform)
        
    def _setup_normalizer(self, train_dataset: RBDataset):
        """Setup data normalizer using training data statistics."""
        print("Computing data normalization statistics...")
        
        # Collect all targets from training data
        all_targets = []
        for i in range(min(len(train_dataset), 100)):  # Use subset for efficiency
            sample = train_dataset[i]
            all_targets.append(sample['targets'])
            
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute statistics
        self.normalizer = DataNormalizer()
        self.normalizer.fit(all_targets)
        
        print(f"Normalization - Mean: {self.normalizer.mean}")
        print(f"Normalization - Std: {self.normalizer.std}")
        
        # Add normalizer to all datasets
        for dataset in self.datasets.values():
            current_transform = dataset.transform
            if current_transform is None:
                dataset.transform = self.normalizer
            else:
                # Chain transforms using a class method (picklable)
                dataset.transform = self._create_chained_transform(current_transform, self.normalizer)
                
    def _create_data_loaders(self):
        """Create data loaders for all datasets."""
        for key, dataset in self.datasets.items():
            shuffle = 'train' in key  # Only shuffle training data
            
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )
            
            self.data_loaders[key] = loader
            print(f"Created data loader for {key}: {len(dataset)} samples, {len(loader)} batches")
            
    def get_dataloader(self, Ra: float, split: str = 'train'):
        """Get data loader for specific Ra and split."""
        key = f"Ra_{Ra:.0e}_{split}"
        if key not in self.data_loaders:
            raise KeyError(f"Data loader not found for {key}. Available: {list(self.data_loaders.keys())}")
        return self.data_loaders[key]
        
    def get_dataset_info(self):
        """Get information about loaded datasets."""
        info = {}
        for key, dataset in self.datasets.items():
            info[key] = {
                'num_samples': len(dataset),
                'spatial_shape': (dataset.H_low, dataset.W_low),
                'high_res_shape': (dataset.H_high, dataset.W_high),
                'Ra': dataset.Ra,
                'Pr': dataset.Pr,
                'dt': dataset.dt,
                'downsample_factors': (dataset.spatial_downsample, dataset.temporal_downsample)
            }
        return info
        
    def create_synthetic_data(self, output_path: str, Ra: float = 1e5, 
                            nx: int = 768, ny: int = 256, nt: int = 600):
        """
        Create synthetic Rayleigh-Bénard data using the original simulation.
        This is a convenience method to generate training data.
        """
        print(f"Generating synthetic RB data: Ra={Ra}, grid={nx}x{ny}, timesteps={nt}")
        
        # Import the original simulation
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rb_simulation import RBNumericalSimulation
        
        # Create simulation
        sim = RBNumericalSimulation(
            nx=nx, ny=ny, Ra=Ra, dt=5e-4,
            save_path=os.path.dirname(output_path)
        )
        
        # Run simulation and collect data
        data_snapshots = []
        times = []
        
        print("Running simulation...")
        for step in range(nt):
            if step % 100 == 0:
                print(f"Step {step}/{nt}")
                
            sim.step(step)
            
            # Collect data every few steps
            if step % 10 == 0:  # Collect every 10 steps
                # Stack fields: [T, p, u, v] -> [H, W, 4]
                snapshot = np.stack([
                    sim.T, sim.p, sim.u, sim.v
                ], axis=-1)
                data_snapshots.append(snapshot)
                times.append(step * sim.dt)
                
        # Convert to array: [T, H, W, 4]
        data_array = np.array(data_snapshots)
        times_array = np.array(times)
        
        print(f"Generated data shape: {data_array.shape}")
        
        # Save to HDF5
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('data', data=data_array)
            f.create_dataset('times', data=times_array)
            f.attrs['Ra'] = Ra
            f.attrs['Pr'] = sim.Pr
            f.attrs['dt'] = sim.dt
            f.attrs['nx'] = nx
            f.attrs['ny'] = ny
            f.attrs['Lx'] = sim.Lx
            f.attrs['Ly'] = sim.Ly
            
        print(f"Saved synthetic data to {output_path}")
        return output_path
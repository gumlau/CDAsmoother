#!/usr/bin/env python3
"""
Regenerate realistic RB training data with improved flow patterns.
"""

import os
import sys
from cdanet.data import RBDataModule

def regenerate_training_data():
    """Regenerate training data with improved flow patterns."""

    print("🔧 Regenerating training data with realistic flow patterns...")

    # Data configuration
    data_dir = './rb_data_numerical'
    output_file = os.path.join(data_dir, 'rb_data_Ra_1e+05.h5')

    # Remove old file if exists
    if os.path.exists(output_file):
        print(f"Removing old data file: {output_file}")
        os.remove(output_file)

    # Create data module
    data_module = RBDataModule(
        data_dir=data_dir,
        spatial_downsample=4,
        temporal_downsample=4,
        clip_length=8,
        batch_size=1,
        num_workers=0,
        pde_points=1000,
        normalize=False
    )

    # Generate new realistic data
    print("Generating realistic Rayleigh-Bénard convection data...")
    synthetic_file = data_module.create_synthetic_data(
        output_path=output_file,
        Ra=1e5,
        nx=256,  # Good resolution
        ny=64,   # Good resolution
        nt=1200  # More timesteps for better training (120 clips)
    )

    print(f"✅ Generated new training data: {synthetic_file}")
    print(f"File size: {os.path.getsize(synthetic_file) / 1024 / 1024:.1f} MB")

    # Test the new data
    print("\n🧪 Testing new data...")
    try:
        data_module.setup([1e5])
        available_loaders = list(data_module.data_loaders.keys())
        print(f"✅ Data loaders: {available_loaders}")

        # Get dataset info
        info = data_module.get_dataset_info()
        for key, dataset_info in info.items():
            print(f"  {key}: {dataset_info['num_samples']} samples")

        print("✅ New data is ready for training!")

    except Exception as e:
        print(f"❌ Error testing new data: {e}")
        return False

    return True

if __name__ == "__main__":
    success = regenerate_training_data()
    if success:
        print("\n🚀 Ready to retrain! Run:")
        print("   python train_paper_config.py")
        print("\n📊 Then visualize results:")
        print("   python visualize_results.py --checkpoint checkpoints/best_model.pth")
    else:
        print("❌ Data regeneration failed!")
        sys.exit(1)
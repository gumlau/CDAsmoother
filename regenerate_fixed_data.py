#!/usr/bin/env python3
"""
Regenerate data with the fixed rb_simulation.py (no more stripes!)
"""

import os
import sys

def regenerate_with_fixed_simulation():
    """Regenerate data using the improved rb_simulation.py"""

    print("🔥 Regenerating data with FIXED rb_simulation.py")
    print("=" * 50)

    # Remove old striped data
    data_dir = './rb_data_numerical'
    data_file = os.path.join(data_dir, 'rb_data_Ra_1e+05.h5')

    if os.path.exists(data_file):
        print(f"🗑️  Removing old striped data: {data_file}")
        old_size = os.path.getsize(data_file) / 1024 / 1024
        print(f"   Old file size: {old_size:.1f} MB")
        os.remove(data_file)

    # Generate new data with fixed function
    print("\n📊 Generating new data with improved 2D structure...")

    from cdanet.data import RBDataModule

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

    # Generate large, high-quality dataset
    synthetic_file = data_module.create_synthetic_data(
        output_path=data_file,
        Ra=1e5,
        nx=256,
        ny=64,
        nt=1600  # Many timesteps for good training
    )

    new_size = os.path.getsize(synthetic_file) / 1024 / 1024
    print(f"✅ Generated new data: {synthetic_file}")
    print(f"📁 New file size: {new_size:.1f} MB")

    # Test the new data
    print("\n🧪 Testing new data quality...")
    try:
        data_module.setup([1e5])
        loaders = list(data_module.data_loaders.keys())
        info = data_module.get_dataset_info()

        print(f"✅ Data loaders: {loaders}")
        for key, dataset_info in info.items():
            print(f"  {key}: {dataset_info['num_samples']} samples")

        total_samples = sum(info[key]['num_samples'] for key in info.keys())
        print(f"📈 Total samples: {total_samples}")

        print("\n🎉 SUCCESS! Fixed data ready for training!")
        print("🚀 Run: python train_paper_config.py")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = regenerate_with_fixed_simulation()
    if not success:
        sys.exit(1)
#!/usr/bin/env python3
"""
Training script using diverse RB data to eliminate horizontal striping
"""

import os
import sys

def main():
    """Train using diverse data to fix horizontal stripe artifacts"""

    print("🎯 Training CDAnet with Diverse Data to Fix Horizontal Stripes")
    print("=" * 60)

    # Use the new diverse dataset
    data_folder = "rb_data_diverse"
    train_data = "rb2d_ra1e+05_consolidated.h5"

    # Check if diverse data exists
    data_path = os.path.join(data_folder, train_data)
    if not os.path.exists(data_path):
        print(f"❌ Diverse dataset not found: {data_path}")
        print("Please generate it first with:")
        print(f"python3 generate_rb_data.py --n_runs 10 --n_samples 30 --Ra 1e5 --save_path {data_folder}")
        return

    print(f"✅ Using diverse dataset: {data_path}")

    # Build training command
    cmd = [
        "python3 train_cdanet_low_memory.py",
        "--epochs 50",
        f"--data_folder {data_folder}",
        f"--train_data {train_data}",
        f"--eval_data {train_data}"
    ]

    training_command = " ".join(cmd)

    print(f"🚀 Training command:")
    print(f"  {training_command}")
    print()
    print("🔧 Expected improvements:")
    print("  - No more horizontal stripes (diverse flow patterns)")
    print("  - Better GPU utilization (2048 sample points)")
    print("  - Physics-informed loss (PDE constraints)")
    print("  - Data augmentation (temporal/spatial)")
    print()

    # Execute training
    print("Starting training...")
    exit_code = os.system(training_command)

    if exit_code == 0:
        print("✅ Training completed successfully!")
        print("🎨 Test visualization with:")
        print(f"   python visualize_results.py --checkpoint ./checkpoints_optimized/checkpoint_epoch_050.pth")
    else:
        print(f"❌ Training failed with exit code: {exit_code}")

if __name__ == "__main__":
    main()
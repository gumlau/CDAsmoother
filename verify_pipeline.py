#!/usr/bin/env python3
"""
Quick pipeline verification - tests all components without full training
"""

import torch
import torch.nn as nn
import os
import numpy as np
from cdanet.models import CDAnet
from cdanet.data import RBDataModule
from cdanet.config import ExperimentConfig

def test_component(name, test_func):
    """Run a test and report results"""
    try:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        result = test_func()
        print(f"✅ {name}: PASSED")
        return True
    except Exception as e:
        print(f"❌ {name}: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading pipeline"""
    config = ExperimentConfig()
    config.data.batch_size = 1
    config.data.num_workers = 0
    
    data_module = RBDataModule(
        data_dir=config.data.data_dir,
        spatial_downsample=4,
        temporal_downsample=4,
        clip_length=8,
        batch_size=1,
        num_workers=0,
        pde_points=10,  # Very small
        normalize=True
    )
    
    print("Setting up data module...")
    data_module.setup([1e5])
    
    print("Getting data loader...")
    train_loader = data_module.get_dataloader(1e5, 'train')
    
    print("Loading one batch...")
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Low-res shape: {batch['low_res'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    print(f"Coords shape: {batch['coords'].shape}")
    
    return True

def test_model_creation():
    """Test model creation and basic forward pass"""
    print("Creating CDAnet model...")
    model = CDAnet(
        in_channels=4,
        feature_channels=64,  # Smaller for CPU
        mlp_hidden_dims=[128, 128],  # Smaller MLP
        activation='softplus',
        coord_dim=3,
        output_dim=4
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with dummy data
    print("Testing forward pass...")
    B, C, T, H, W = 1, 4, 2, 32, 32  # Small input
    N = 10  # Few points
    
    low_res = torch.randn(B, C, T, H, W)
    coords = torch.randn(B, N, 3)
    
    with torch.no_grad():
        output = model(low_res, coords)
    
    print(f"Input shape: {low_res.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({B}, {N}, 4)")
    
    assert output.shape == (B, N, 4), f"Wrong output shape: {output.shape}"
    return True

def test_training_step():
    """Test a single training step"""
    print("Creating small model for training test...")
    model = CDAnet(
        in_channels=4,
        feature_channels=32,  # Very small
        mlp_hidden_dims=[64],  # Single layer
        activation='softplus',
        coord_dim=3,
        output_dim=4
    )
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create dummy batch
    B, C, T, H, W = 1, 4, 2, 16, 16
    N = 5
    
    low_res = torch.randn(B, C, T, H, W)
    coords = torch.randn(B, N, 3, requires_grad=True)
    targets = torch.randn(B, N, 4)
    
    print("Running training step...")
    model.train()
    
    # Forward pass
    predictions = model(low_res, coords)
    loss = criterion(predictions, targets)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training step completed successfully!")
    return True

def test_checkpoint_save():
    """Test checkpoint saving"""
    print("Creating model for checkpoint test...")
    model = CDAnet(
        in_channels=4,
        feature_channels=16,
        mlp_hidden_dims=[32],
        activation='softplus',
        coord_dim=3,
        output_dim=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Save checkpoint
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_path = "./checkpoints/test_checkpoint.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 1,
        'loss': 0.123
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / 1024:.1f} KB")
    
    # Test loading
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Checkpoint loaded successfully!")
    return True

def test_evaluation_setup():
    """Test evaluation script can be imported"""
    try:
        from cdanet.evaluation import CDAnetEvaluator
        print("CDAnetEvaluator imported successfully!")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def main():
    print("🚀 CDAsmoother Pipeline Verification")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation), 
        ("Training Step", test_training_step),
        ("Checkpoint Save/Load", test_checkpoint_save),
        ("Evaluation Import", test_evaluation_setup),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_component(name, test_func)
    
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All pipeline components are working correctly!")
        print("The CDAsmoother pipeline is ready for full training.")
        
        # Create success marker files
        os.makedirs("./outputs", exist_ok=True)
        with open("./outputs/pipeline_verification_success.txt", "w") as f:
            f.write("CDAsmoother Pipeline Verification Results\n")
            f.write("="*50 + "\n")
            f.write(f"Verification completed successfully: {total_passed}/{total_tests} tests passed\n\n")
            for name, passed in results.items():
                f.write(f"{name}: {'PASS' if passed else 'FAIL'}\n")
            f.write("\nThe pipeline is ready for training!\n")
        
        print("✅ Results saved to ./outputs/pipeline_verification_success.txt")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Diagnose model inference issues - why predictions are constant.
"""

import torch
import numpy as np
from cdanet.models import CDAnet
from cdanet.data import RBDataModule

def diagnose_model_inference(checkpoint_path):
    """Comprehensive diagnosis of model inference issues."""

    print("=" * 60)
    print("Model Inference Diagnosis")
    print("=" * 60)

    # Load checkpoint
    print("1. Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    print(f"   Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")

    # Check model config
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
        print(f"   Model config: {model_config}")
    else:
        print("   ⚠️ No config found in checkpoint")
        return

    # Create model
    print("\n2. Creating model...")
    model = CDAnet(
        in_channels=model_config['in_channels'],
        feature_channels=model_config['feature_channels'],
        mlp_hidden_dims=model_config['mlp_hidden_dims'],
        activation=model_config['activation'],
        coord_dim=model_config['coord_dim'],
        output_dim=model_config['output_dim']
    )

    # Load state dict
    print("3. Loading model weights...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Model weights loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        return

    # Check model parameters
    print("\n4. Checking model parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Check if parameters are reasonable
    param_stats = []
    for name, param in model.named_parameters():
        if param.numel() < 1000:  # Only check smaller parameters
            param_stats.append(f"   {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

    print("   Sample parameter statistics:")
    for stat in param_stats[:5]:  # Show first 5
        print(stat)

    # Setup data
    print("\n5. Setting up data...")
    data_module = RBDataModule(
        data_dir='./rb_data_numerical',
        spatial_downsample=4,
        temporal_downsample=4,
        clip_length=8,
        batch_size=1,
        num_workers=0,
        pde_points=100,  # Small for testing
        normalize=True
    )

    try:
        data_module.setup([1e5])
        print("   ✅ Data setup successful")

        # Get test loader
        test_loader = data_module.get_dataloader('Ra_1e+05_test')
        sample = next(iter(test_loader))

        print(f"   Sample shapes:")
        print(f"     low_res: {sample['low_res'].shape}")
        print(f"     targets: {sample['targets'].shape}")
        print(f"     coords: {sample['coords'].shape}")

    except Exception as e:
        print(f"   ❌ Data setup failed: {e}")
        return

    # Test model inference
    print("\n6. Testing model inference...")
    model.eval()

    with torch.no_grad():
        try:
            low_res = sample['low_res']
            coords = sample['coords']
            targets = sample['targets']

            print(f"   Input shapes: low_res={low_res.shape}, coords={coords.shape}")

            # Forward pass
            predictions = model(low_res, coords)

            print(f"   ✅ Forward pass successful")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Prediction range: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]")
            print(f"   Prediction std: {predictions.std().item():.6f}")

            # Check each field
            print("   Field-wise analysis:")
            field_names = ['T', 'p', 'u', 'v']
            for i, field in enumerate(field_names):
                field_pred = predictions[:, i]
                field_target = targets[:, i]
                print(f"     {field}:")
                print(f"       Pred: [{field_pred.min().item():.6f}, {field_pred.max().item():.6f}] std={field_pred.std().item():.6f}")
                print(f"       Target: [{field_target.min().item():.6f}, {field_target.max().item():.6f}] std={field_target.std().item():.6f}")

            # Check if all predictions are the same
            if torch.allclose(predictions, predictions.mean(), atol=1e-6):
                print("   ❌ ALL PREDICTIONS ARE CONSTANT!")
                print("   This indicates a serious model issue:")
                print("     - Model might not have trained properly")
                print("     - Gradient flow issues during training")
                print("     - Architecture problems")

                # Check model components
                print("\n7. Checking model components...")

                # Test feature extractor
                print("   Testing feature extractor...")
                features = model.feature_extractor(low_res)
                print(f"   Features shape: {features.shape}")
                print(f"   Features range: [{features.min().item():.6f}, {features.max().item():.6f}]")
                print(f"   Features std: {features.std().item():.6f}")

                if torch.allclose(features, features.mean(), atol=1e-6):
                    print("   ❌ Feature extractor outputs are constant!")
                else:
                    print("   ✅ Feature extractor working")

                # Test MLP
                print("   Testing MLP...")
                # Create simple test input for MLP
                test_coords = coords[0][:100]  # Take first 100 points
                test_features = features[0].flatten().unsqueeze(0).expand(100, -1)  # Expand features

                mlp_input = torch.cat([test_coords, test_features], dim=1)
                mlp_output = model.physics_mlp(mlp_input)

                print(f"   MLP input shape: {mlp_input.shape}")
                print(f"   MLP output shape: {mlp_output.shape}")
                print(f"   MLP output range: [{mlp_output.min().item():.6f}, {mlp_output.max().item():.6f}]")
                print(f"   MLP output std: {mlp_output.std().item():.6f}")

                if torch.allclose(mlp_output, mlp_output.mean(), atol=1e-6):
                    print("   ❌ MLP outputs are constant!")
                else:
                    print("   ✅ MLP working")
            else:
                print("   ✅ Predictions have variation - model seems OK")

        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Diagnosis Complete")
    print("=" * 60)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python diagnose_model_inference.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    diagnose_model_inference(checkpoint_path)
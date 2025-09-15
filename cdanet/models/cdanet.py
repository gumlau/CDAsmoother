"""
CDAnet: Complete Physics-Informed Deep Neural Network for fluid dynamics.
Combines 3D U-Net feature extractor with physics-informed MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet3d import UNet3D
from .mlp import PhysicsInformedMLP


class CDAnet(nn.Module):
    """
    Complete CDAnet architecture combining feature extraction and physics-informed prediction.
    
    Args:
        in_channels: Number of input channels (4 for T, p, u, v)
        feature_channels: Number of feature channels from U-Net
        mlp_hidden_dims: List of hidden dimensions for MLP
        activation: Activation function for MLP
        coord_dim: Dimension of coordinates (3 for x, y, t)
        output_dim: Number of output variables (4 for T, p, u, v)
        base_channels: Base number of channels for UNet3D first layer
    """
    
    def __init__(self, in_channels=4, feature_channels=256, mlp_hidden_dims=[512, 512, 512, 512],
                 activation='softplus', coord_dim=3, output_dim=4, base_channels=32, **kwargs):
        super(CDAnet, self).__init__()
        
        # Feature extractor (3D U-Net)
        self.feature_extractor = UNet3D(
            in_channels=in_channels,
            feature_channels=feature_channels,
            base_channels=base_channels
        )
        
        # Physics-informed MLP
        self.mlp = PhysicsInformedMLP(
            feature_dim=feature_channels,
            coord_dim=coord_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dim,
            activation=activation
        )
        
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        
    def forward(self, low_res_input, coords):
        """
        Forward pass through CDAnet.
        
        Args:
            low_res_input: Low-resolution spatio-temporal clips [B, C, T, H, W]
            coords: Spatio-temporal coordinates for high-res prediction [B, N, 3]
            
        Returns:
            outputs: High-resolution predictions [B, N, 4] (T, p, u, v)
        """
        # Extract features using 3D U-Net
        features_3d = self.feature_extractor(low_res_input)  # [B, feature_channels, T, H, W]
        
        # Interpolate features to coordinate locations
        features = self._interpolate_features(features_3d, coords)  # [B, N, feature_channels]
        
        # Predict high-resolution fields using MLP
        outputs = self.mlp(features, coords)
        
        return outputs
    
    def forward_with_derivatives(self, low_res_input, coords):
        """
        Forward pass with derivative computation for PDE loss.
        
        Args:
            low_res_input: Low-resolution input clips [B, C, T, H, W]
            coords: Coordinates for derivative computation [B, N, 3]
            
        Returns:
            outputs: Predicted fields [B, N, 4]
            derivatives: Dictionary of partial derivatives
        """
        # Extract features
        features_3d = self.feature_extractor(low_res_input)
        features = self._interpolate_features(features_3d, coords)
        
        # Compute predictions and derivatives
        outputs, derivatives = self.mlp.compute_derivatives(features, coords)
        
        return outputs, derivatives
    
    def _interpolate_features(self, features_3d, coords):
        """
        Interpolate 3D features to specific coordinate locations.
        
        Args:
            features_3d: Feature tensor [B, C, T, H, W]
            coords: Target coordinates [B, N, 3] (normalized to [-1, 1])
            
        Returns:
            interpolated_features: Features at target coordinates [B, N, C]
        """
        B, C, T, H, W = features_3d.shape
        B_coords, N, _ = coords.shape
        
        # Ensure batch sizes match
        assert B == B_coords, f"Batch size mismatch: features {B}, coords {B_coords}"
        
        # Reshape coordinates for grid_sample: [B, 1, 1, N, 3]
        coords_reshaped = coords.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]
        
        # Transpose features for grid_sample: [B, C, T, H, W] -> [B, C, W, H, T]
        features_transposed = features_3d.permute(0, 1, 4, 3, 2)
        
        # Use grid_sample for 3D interpolation
        # coords should be in order (x, y, t) corresponding to (W, H, T)
        interpolated = F.grid_sample(
            features_transposed, 
            coords_reshaped, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )  # [B, C, 1, 1, N]
        
        # Reshape to [B, N, C]
        interpolated_features = interpolated.squeeze(2).squeeze(2).transpose(1, 2)
        
        return interpolated_features
    
    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_info(self):
        """Print model architecture information."""
        print("=" * 50)
        print("CDAnet Model Architecture")
        print("=" * 50)
        print(f"Total parameters: {self.get_num_parameters():,}")
        print()
        print("Feature Extractor (3D U-Net):")
        unet_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        print(f"  Parameters: {unet_params:,}")
        print()
        print("Physics-Informed MLP:")
        mlp_params = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"  Parameters: {mlp_params:,}")
        print("=" * 50)
"""
CDAnet: Complete Physics-Informed Deep Neural Network for fluid dynamics.
Combines 3D U-Net feature extractor with implicit neural network.
Based on the reference sourcecodeCDAnet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet3d_reference import UNet3d
from .implicit_net import ImNet, get_activation
from .local_implicit_grid import query_local_implicit_grid


class CDAnet(nn.Module):
    """
    Complete CDAnet architecture combining UNet3D feature extraction and ImNet prediction.
    Based on the reference sourcecodeCDAnet implementation.

    Args:
        in_channels: Number of input channels (4 for T, p, u, v)
        feature_channels: Number of latent feature channels from U-Net
        mlp_hidden_dims: List with 2 elements [latent_dims, imnet_nf] for backward compatibility
        activation: Activation function for ImNet
        coord_dim: Dimension of coordinates (3 for x, y, t)
        output_dim: Number of output variables (4 for T, p, u, v)
        igres: Input grid resolution (will be determined from data)
        unet_nf: Base number of features for UNet
        unet_mf: Max number of features for UNet
    """

    def __init__(self, in_channels=4, feature_channels=128, mlp_hidden_dims=[128, 256],
                 activation='softplus', coord_dim=3, output_dim=4,
                 igres=(8, 32, 64), unet_nf=16, unet_mf=512, **kwargs):
        super(CDAnet, self).__init__()

        # Parse parameters for backward compatibility
        if len(mlp_hidden_dims) >= 2:
            lat_dims = mlp_hidden_dims[0]  # latent dimensions
            imnet_nf = mlp_hidden_dims[1]  # ImNet width
        else:
            lat_dims = feature_channels
            imnet_nf = 256

        # Feature extractor (3D U-Net) - matches reference architecture
        self.feature_extractor = UNet3d(
            in_features=in_channels,
            out_features=lat_dims,
            igres=igres,
            nf=unet_nf,
            mf=unet_mf
        )

        # Implicit neural network - matches reference architecture
        activation_fn = get_activation(activation)
        self.implicit_net = ImNet(
            dim=coord_dim,
            in_features=lat_dims,
            out_features=output_dim,
            nf=imnet_nf,
            activation=activation_fn
        )

        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.lat_dims = lat_dims
        
    def forward(self, low_res_input, coords):
        """
        Forward pass through CDAnet following reference architecture.

        Args:
            low_res_input: Low-resolution input tensor [batch, channels, T, H, W]
            coords: Query coordinates [batch, num_points, coord_dim]

        Returns:
            predictions: Predicted values at query coordinates [batch, num_points, output_dim]
        """
        # Step 1: Extract latent features using UNet3d
        latent_grid = self.feature_extractor(low_res_input)  # [batch, lat_dims, T, H, W]

        # Step 2: Permute for implicit grid query - reference format
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, H, W, lat_dims]

        # Step 3: Query the implicit grid using reference method
        # Define domain bounds (normalized coordinates -1 to 1)
        xmin = torch.tensor([-1.0, -1.0, -1.0], device=coords.device)
        xmax = torch.tensor([1.0, 1.0, 1.0], device=coords.device)

        # Use the reference query method
        predictions = query_local_implicit_grid(self.implicit_net, latent_grid, coords, xmin, xmax)

        return predictions

    def get_latent_grid(self, low_res_input):
        """Get latent grid representation (useful for visualization/debugging)."""
        latent_grid = self.feature_extractor(low_res_input)
        return latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, H, W, lat_dims]

    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_model_info(self):
        """Print model architecture information."""
        print("=" * 50)
        print("CDAnet Model Architecture (Reference-based)")
        print("=" * 50)
        print(f"Total parameters: {self.get_num_parameters():,}")
        print()
        print("Feature Extractor (UNet3d):")
        unet_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        print(f"  Parameters: {unet_params:,}")
        print()
        print("Implicit Network (ImNet):")
        imnet_params = sum(p.numel() for p in self.implicit_net.parameters() if p.requires_grad)
        print(f"  Parameters: {imnet_params:,}")
        print("=" * 50)
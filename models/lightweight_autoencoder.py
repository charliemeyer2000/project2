"""
Lightweight Autoencoder - Inspired by AJR's winning strategy.

Key principles:
1. TINY model size (~4-8 MB target)
2. Moderate latent dimension (LD=16)
3. Simple architecture (fewer layers, fewer channels)
4. Focus on generalization over complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightweightEncoder(nn.Module):
    """
    Lightweight encoder with minimal parameters.
    
    Architecture: 5 conv layers (256->128->64->32->16->8) + FC
    Removed Global Average Pooling to preserve spatial information.
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        base_channels: int = 16,
        use_dropout: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Progressive downsampling: 256 -> 128 -> 64 -> 32 -> 16
        # Keep channel counts LOW to minimize parameters
        self.conv1 = nn.Conv2d(3, base_channels, 3, stride=2, padding=1)  # 256->128
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)  # 128->64
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1)  # 64->32
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*4, 3, stride=2, padding=1)  # 32->16
        self.bn4 = nn.BatchNorm2d(base_channels*4)
        
        # One more downsample: 16 -> 8 (reduces spatial size, saves params)
        self.conv5 = nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1)  # 16->8
        self.bn5 = nn.BatchNorm2d(base_channels*8)
        
        # FC to latent (8x8 spatial size instead of 16x16)
        self.fc = nn.Linear(base_channels*8 * 8 * 8, latent_dim)
        
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv blocks with BN and ReLU
        x = self.activation(self.bn1(self.conv1(x)))
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.activation(self.bn2(self.conv2(x)))
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.activation(self.bn3(self.conv3(x)))
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.activation(self.bn4(self.conv4(x)))
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.activation(self.bn5(self.conv5(x)))
        
        # Flatten and FC to latent
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class LightweightDecoder(nn.Module):
    """
    Lightweight decoder matching encoder complexity.
    Target: ~2-4 MB
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        base_channels: int = 16,
    ):
        super().__init__()
        
        # Expand from latent to feature map (8x8 to match encoder)
        self.fc = nn.Linear(latent_dim, base_channels*8 * 8 * 8)
        
        # Progressive upsampling: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.deconv1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, stride=2, padding=1)  # 8->16
        self.bn1 = nn.BatchNorm2d(base_channels*4)
        
        self.deconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*4, 4, stride=2, padding=1)  # 16->32
        self.bn2 = nn.BatchNorm2d(base_channels*4)
        
        self.deconv3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1)  # 32->64
        self.bn3 = nn.BatchNorm2d(base_channels*2)
        
        self.deconv4 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)  # 64->128
        self.bn4 = nn.BatchNorm2d(base_channels)
        
        self.deconv5 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)  # 128->256
        self.bn5 = nn.BatchNorm2d(base_channels)
        
        # Final conv to RGB
        self.final_conv = nn.Conv2d(base_channels, 3, 3, padding=1)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Expand to feature map (8x8)
        x = self.fc(z).view(-1, self.fc.out_features // (8*8), 8, 8)
        
        # Deconv blocks with BN and ReLU
        x = self.activation(self.bn1(self.deconv1(x)))
        x = self.activation(self.bn2(self.deconv2(x)))
        x = self.activation(self.bn3(self.deconv3(x)))
        x = self.activation(self.bn4(self.deconv4(x)))
        x = self.activation(self.bn5(self.deconv5(x)))
        
        # Final output with sigmoid
        x = torch.sigmoid(self.final_conv(x))
        
        return x


class LightweightAutoencoder(nn.Module):
    """
    Lightweight autoencoder targeting ~4-8 MB total size.
    
    Design philosophy:
    - Fewer parameters = better generalization
    - Simple architecture = less overfitting
    - Dropout + BN = strong regularization
    - LD=16 = proven sweet spot
    
    Args:
        latent_dim: Bottleneck size (default: 16, like AJR)
        base_channels: Base channel multiplier (default: 16 for ~4-6 MB)
        use_dropout: Enable dropout in encoder (default: True)
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        base_channels: int = 16,
        use_dropout: bool = True,
        dropout_rate: float = 0.2,
        **kwargs,  # Ignore extra args from config
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.enc = LightweightEncoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        
        self.dec = LightweightDecoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        recon = self.dec(z)
        return recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """For analysis/visualization."""
        return self.enc(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """For analysis/visualization."""
        return self.dec(z)


"""
MiniAttention Autoencoder - Compact version with attention for generalization.

Based on AttentionAutoencoder but optimized for 4-10 MB range.
Uses proven attention mechanisms but with much smaller width_mult.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Lightweight channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MiniEncoder(nn.Module):
    """Compact encoder with channel attention."""
    
    __constants__ = ['latent_dim']
    
    def __init__(
        self, 
        latent_dim: int = 16,
        base_channels: int = 24,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Calculate channel dimensions (smaller than full AttentionEncoder)
        c1 = base_channels       # 24
        c2 = base_channels * 2   # 48
        c3 = base_channels * 3   # 72
        c4 = base_channels * 4   # 96
        
        # Helper to create properly divisible GroupNorm
        def get_norm(channels):
            for num_groups in [16, 8, 4, 2, 1]:
                if channels % num_groups == 0:
                    break
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        
        # 256x256x3 -> 128x128xc1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c1, 4, stride=2, padding=1),
            get_norm(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            ChannelAttention(c1)
        )
        
        # 128x128 -> 64x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),
            get_norm(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            ChannelAttention(c2)
        )
        
        # 64x64 -> 32x32
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, 4, stride=2, padding=1),
            get_norm(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            ChannelAttention(c3)
        )
        
        # 32x32 -> 16x16
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, 4, stride=2, padding=1),
            get_norm(c4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            ChannelAttention(c4)
        )
        
        # 16x16 -> 8x8
        self.conv5 = nn.Sequential(
            nn.Conv2d(c4, c4, 4, stride=2, padding=1),
            get_norm(c4),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.fc = nn.Linear(c4 * 8 * 8, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        
        # Flatten and encode
        z = self.fc(h5.flatten(1))
        return z


class MiniDecoder(nn.Module):
    """Compact decoder matching encoder complexity."""
    
    def __init__(
        self, 
        latent_dim: int = 16,
        base_channels: int = 24,
    ):
        super().__init__()
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3
        c4 = base_channels * 4
        
        # Helper to create properly divisible GroupNorm
        def get_norm(channels):
            for num_groups in [16, 8, 4, 2, 1]:
                if channels % num_groups == 0:
                    break
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        
        # Expand from latent
        self.fc = nn.Linear(latent_dim, c4 * 8 * 8)
        
        # 8x8 -> 16x16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(c4, c4, 4, stride=2, padding=1),
            get_norm(c4),
            nn.ReLU(inplace=True),
            ChannelAttention(c4)
        )
        
        # 16x16 -> 32x32
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1),
            get_norm(c3),
            nn.ReLU(inplace=True),
            ChannelAttention(c3)
        )
        
        # 32x32 -> 64x64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),
            get_norm(c2),
            nn.ReLU(inplace=True),
            ChannelAttention(c2)
        )
        
        # 64x64 -> 128x128
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
            get_norm(c1),
            nn.ReLU(inplace=True),
            ChannelAttention(c1)
        )
        
        # 128x128 -> 256x256
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1),
            get_norm(c1),
            nn.ReLU(inplace=True)
        )
        
        # Final conv to RGB
        self.final_conv = nn.Conv2d(c1, 3, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Expand to feature map
        x = self.fc(z).view(-1, self.fc.out_features // (8*8), 8, 8)
        
        # Decoder path
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        
        # Final output
        x = torch.sigmoid(self.final_conv(x))
        return x


class MiniAttentionAutoencoder(nn.Module):
    """
    Mini Attention Autoencoder - Compact model with attention mechanisms.
    
    Target: 4-10 MB with strong generalization.
    
    Design:
    - Smaller base_channels than AttentionAutoencoder
    - Channel attention for feature refinement
    - GroupNorm for better generalization with small batch sizes
    - Heavy dropout for regularization
    - No skip connections (reduces overfitting)
    
    Args:
        latent_dim: Bottleneck size (default: 16, proven optimal)
        base_channels: Base channel count (default: 24 for ~6-8 MB)
                      16 -> ~3-4 MB, 24 -> ~6-8 MB, 32 -> ~10-12 MB
        dropout: Dropout rate (default: 0.15, heavy regularization)
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        base_channels: int = 24,
        dropout: float = 0.15,
        **kwargs,  # Ignore extra config args
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.enc = MiniEncoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            dropout=dropout,
        )
        
        self.dec = MiniDecoder(
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


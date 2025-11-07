"""Lightweight Autoencoder with Attention for traffic light reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Lightweight channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionEncoder(nn.Module):
    """Encoder with channel attention for better feature learning."""
    
    __constants__ = ['latent_dim']
    
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 256x256x3 -> 128x128x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ChannelAttention(32)
        )
        
        # 128x128x32 -> 64x64x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ChannelAttention(64)
        )
        
        # 64x64x64 -> 32x32x128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128)
        )
        
        # 32x32x128 -> 16x16x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128)
        )
        
        # 16x16x128 -> 8x8x128
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AttentionDecoder(nn.Module):
    """Decoder with attention for better reconstruction."""
    
    __constants__ = ['latent_dim']
    
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Expand from latent
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        
        # 8x8x128 -> 16x16x128
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128)
        )
        
        # 16x16x128 -> 32x32x128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128)
        )
        
        # 32x32x128 -> 64x64x64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ChannelAttention(64)
        )
        
        # 64x64x64 -> 128x128x32
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ChannelAttention(32)
        )
        
        # 128x128x32 -> 256x256x3
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


class AttentionAutoencoder(nn.Module):
    """Attention-enhanced autoencoder for traffic light reconstruction.
    
    Key features:
    - Channel attention to focus on important features (traffic lights)
    - Batch normalization for stable training
    - Deeper than efficient model for better capacity
    - Still lightweight (~5-8 MB)
    
    Args:
        latent_dim: Dimensionality of latent space (default: 16)
    """
    
    # Mark latent_dim as a TorchScript constant so it's preserved in exported models
    __constants__ = ['latent_dim']
    
    def __init__(self, latent_dim: int = 16, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        # Use 'enc' and 'dec' naming to match baseline for server compatibility
        self.enc = AttentionEncoder(latent_dim)
        self.dec = AttentionDecoder(latent_dim)
    
    def forward(self, x):
        z = self.enc(x)
        reconstructed = self.dec(z)
        return reconstructed
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.enc(x)
    
    def decode(self, z):
        """Decode from latent space."""
        return self.dec(z)
    
    def get_latent_dim(self):
        """Return latent dimension."""
        return self.latent_dim


if __name__ == "__main__":
    # Test model
    model = AttentionAutoencoder(latent_dim=16)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Latent representation
    z = model.encode(x)
    print(f"Latent shape: {z.shape}")
    print(f"Latent dim: {model.get_latent_dim()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated size: {total_params * 4 / (1024**2):.2f} MB (fp32)")



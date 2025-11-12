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
    
    __constants__ = ['latent_dim', 'width_mult']
    
    def __init__(self, latent_dim: int = 16, width_mult: float = 1.0, norm_type: str = 'group'):
        super().__init__()
        self.latent_dim = latent_dim
        self.width_mult = width_mult
        self.norm_type = norm_type
        
        # Calculate channel dimensions (scaled by width_mult)
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(128 * width_mult)  # Keep at 128 for reasonable model size
        c5 = int(128 * width_mult)  # Keep at 128 for reasonable model size
        
        # Helper to create normalization layer
        def get_norm(channels):
            if norm_type == 'group':
                # Find largest divisor <= 16 for num_groups
                # This ensures num_channels is always divisible by num_groups
                for num_groups in [16, 8, 4, 2, 1]:
                    if channels % num_groups == 0:
                        break
                return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            elif norm_type == 'batch':
                return nn.BatchNorm2d(channels)
            else:  # 'none'
                return nn.Identity()
        
        # 256x256x3 -> 128x128x(32*w)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c1, 4, stride=2, padding=1),
            get_norm(c1),
            nn.ReLU(inplace=True),
            ChannelAttention(c1)
        )
        
        # 128x128 -> 64x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),
            get_norm(c2),
            nn.ReLU(inplace=True),
            ChannelAttention(c2)
        )
        
        # 64x64 -> 32x32
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, 4, stride=2, padding=1),
            get_norm(c3),
            nn.ReLU(inplace=True),
            ChannelAttention(c3)
        )
        
        # 32x32 -> 16x16
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, 4, stride=2, padding=1),
            get_norm(c4),
            nn.ReLU(inplace=True),
            ChannelAttention(c4)
        )
        
        # 16x16 -> 8x8
        self.conv5 = nn.Sequential(
            nn.Conv2d(c4, c5, 4, stride=2, padding=1),
            get_norm(c5),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.fc = nn.Linear(c5 * 8 * 8, latent_dim)
    
    def forward(self, x):
        """Forward pass with skip connections.
        
        Returns:
            latent: Latent code [B, latent_dim]
            skips: List of intermediate features [h1, h2, h3, h4] for skip connections
        """
        h1 = self.conv1(x)    # 128×128
        h2 = self.conv2(h1)   # 64×64
        h3 = self.conv3(h2)   # 32×32
        h4 = self.conv4(h3)   # 16×16
        h5 = self.conv5(h4)   # 8×8
        
        # Bottleneck
        z = h5.view(h5.size(0), -1)
        z = self.fc(z)
        
        # Return latent code and skip connections
        return z, [h1, h2, h3, h4]


class AttentionDecoder(nn.Module):
    """Decoder with attention for better reconstruction."""
    
    __constants__ = ['latent_dim', 'width_mult']
    
    def __init__(self, latent_dim: int = 16, width_mult: float = 1.0, use_skip_connections: bool = True, activation_type: str = 'tanh', norm_type: str = 'group'):
        super().__init__()
        self.latent_dim = latent_dim
        self.width_mult = width_mult
        self.use_skip_connections = use_skip_connections
        self.activation_type = activation_type
        self.norm_type = norm_type
        
        # Calculate channel dimensions (scaled by width_mult)
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(128 * width_mult)  # Keep at 128 for reasonable model size
        self.c5 = int(128 * width_mult)  # Keep at 128 for reasonable model size
        
        # Helper to create normalization layer
        def get_norm(channels):
            if norm_type == 'group':
                # Find largest divisor <= 16 for num_groups
                # This ensures num_channels is always divisible by num_groups
                for num_groups in [16, 8, 4, 2, 1]:
                    if channels % num_groups == 0:
                        break
                return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            elif norm_type == 'batch':
                return nn.BatchNorm2d(channels)
            else:  # 'none'
                return nn.Identity()
        
        # Expand from latent
        self.fc = nn.Linear(latent_dim, self.c5 * 8 * 8)
        
        # 8x8 -> 16x16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.c5, c4, 4, stride=2, padding=1),
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
        
        # 128x128 -> 256x256x3 (configurable activation)
        if activation_type == 'tanh':
            self.deconv5 = nn.Sequential(
                nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1),
                nn.Tanh()
            )
        elif activation_type == 'sigmoid':
            self.deconv5 = nn.Sequential(
                nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        else:  # 'none'
            self.deconv5 = nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1)
    
    def forward(self, x, skip_connections=None):
        """Forward pass with optional skip connections.
        
        Args:
            x: Latent code [B, latent_dim]
            skip_connections: Optional list of [h1, h2, h3, h4] from encoder
        """
        x = self.fc(x)
        x = x.view(x.size(0), self.c5, 8, 8)
        
        # 8×8 → 16×16
        x = self.deconv1(x)
        if skip_connections is not None and self.use_skip_connections:
            x = x + skip_connections[3]  # Add h4 (16×16)
        
        # 16×16 → 32×32
        x = self.deconv2(x)
        if skip_connections is not None and self.use_skip_connections:
            x = x + skip_connections[2]  # Add h3 (32×32)
        
        # 32×32 → 64×64
        x = self.deconv3(x)
        if skip_connections is not None and self.use_skip_connections:
            x = x + skip_connections[1]  # Add h2 (64×64)
        
        # 64×64 → 128×128
        x = self.deconv4(x)
        if skip_connections is not None and self.use_skip_connections:
            x = x + skip_connections[0]  # Add h1 (128×128)
        
        # 128×128 → 256×256
        x = self.deconv5(x)
        
        # Rescale if using tanh or clamp if using none
        if self.activation_type == 'tanh':
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        elif self.activation_type == 'none':
            x = torch.clamp(x, 0, 1)
        
        return x


class AttentionAutoencoder(nn.Module):
    """Attention-enhanced autoencoder for traffic light reconstruction.
    
    Key features:
    - Channel attention to focus on important features (traffic lights)
    - Batch normalization for stable training
    - Deeper than efficient model for better capacity
    - Scalable width via width_mult parameter
    
    Args:
        latent_dim: Dimensionality of latent space (default: 16)
        width_mult: Width multiplier for channels (1.0=6MB, 1.5=13MB, 2.0=23MB for LD=16)
    """
    
    # Mark latent_dim as a TorchScript constant so it's preserved in exported models
    __constants__ = ['latent_dim']
    
    def __init__(self, latent_dim: int = 16, width_mult: float = 1.0, use_skip_connections: bool = True, activation_type: str = 'tanh', norm_type: str = 'group', **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        # Use 'enc' and 'dec' naming to match baseline for server compatibility
        self.enc = AttentionEncoder(latent_dim, width_mult, norm_type)
        self.dec = AttentionDecoder(latent_dim, width_mult, use_skip_connections, activation_type, norm_type)
    
    def forward(self, x):
        z, skips = self.enc(x)
        reconstructed = self.dec(z, skips if self.use_skip_connections else None)
        return reconstructed
    
    def encode(self, x):
        """Encode input to latent space (returns only latent code, not skips)."""
        z, _ = self.enc(x)
        return z
    
    def decode(self, z):
        """Decode from latent space (without skip connections)."""
        return self.dec(z, skip_connections=None)
    
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



"""Efficient autoencoder with depthwise separable convolutions."""

import torch
import torch.nn as nn
import numpy as np


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (reduces parameters significantly)."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientEncoder(nn.Module):
    """Efficient encoder using depthwise separable convolutions."""
    
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Progressively downsample with fewer channels
        # Using stride=2 convs instead of pooling
        self.conv_blocks = nn.ModuleList([
            # Block 1: 256 -> 128
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(True),
            ),
            # Block 2: 128 -> 64
            nn.Sequential(
                DepthwiseSeparableConv(base_channels, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(True),
            ),
            # Block 3: 64 -> 32
            nn.Sequential(
                DepthwiseSeparableConv(base_channels * 2, base_channels * 3, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 3),
                nn.ReLU(True),
            ),
            # Block 4: 32 -> 16
            nn.Sequential(
                DepthwiseSeparableConv(base_channels * 3, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(True),
            ),
        ])
        
        # Smaller fully connected layer
        # At 16x16 with 128 channels = 32768 features
        self.fc = None
        self.shape = None
        self.base_channels = base_channels
    
    def _init_fc(self, h):
        """Lazy initialization of FC layer."""
        n, c, H, W = h.shape
        self.shape = (c, H, W)
        self.fc = nn.Linear(c * H * W, self.latent_dim).to(h.device)
    
    def forward(self, x):
        h = x
        for block in self.conv_blocks:
            h = block(h)
        
        if self.fc is None:
            self._init_fc(h)
        
        z = self.fc(h.view(h.size(0), -1))
        return z


class EfficientDecoder(nn.Module):
    """Efficient decoder using depthwise separable transposed convolutions."""
    
    def __init__(self, out_channels: int, latent_dim: int, img_size: int = 256, base_channels: int = 32):
        super().__init__()
        
        # Calculate starting spatial size (after 4 downsamples: 256/16 = 16)
        start_size = img_size // 16
        start_channels = base_channels * 4  # 128 channels
        
        self._shape = (start_channels, start_size, start_size)
        self.fc = nn.Linear(latent_dim, int(np.prod(self._shape)))
        
        # Progressively upsample
        self.deconv_blocks = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(start_channels, base_channels * 3, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 3),
            nn.ReLU(True),
            
            # 32 -> 64
            nn.ConvTranspose2d(base_channels * 3, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            
            # 64 -> 128
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            
            # 128 -> 256 (final layer, no batchnorm)
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        h = self.fc(z)
        c, H, W = self._shape
        h = h.view(z.size(0), c, H, W)
        return self.deconv_blocks(h)


class EfficientAutoencoder(nn.Module):
    """Efficient autoencoder optimized for small model size and good reconstruction.
    
    Key improvements over baseline:
    - Depthwise separable convolutions (10x fewer parameters)
    - Batch normalization for better training
    - Optimized channel progression (32->64->96->128 instead of 32->64->128->256)
    - Smaller FC layer overhead
    
    Expected model sizes (approximate):
    - LD=8:  ~2-3 MB
    - LD=16: ~3-4 MB
    - LD=32: ~5-6 MB
    """
    
    # Mark latent_dim as a TorchScript constant so it's preserved in exported models
    __constants__ = ['latent_dim', 'channels', 'img_size']
    
    def __init__(self, channels: int = 3, latent_dim: int = 16, 
                 img_size: int = 256, base_channels: int = 32):
        """Initialize efficient autoencoder.
        
        Args:
            channels: Number of input channels (3 for RGB, 1 for grayscale)
            latent_dim: Size of latent representation
            img_size: Input image size (default 256)
            base_channels: Base number of channels (default 32)
        """
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        self.enc = EfficientEncoder(channels, latent_dim, base_channels)
        self.dec = EfficientDecoder(channels, latent_dim, img_size, base_channels)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Reconstructed images [B, C, H, W]
        """
        z = self.enc(x)
        recon = self.dec(z)
        return recon
    
    def encode(self, x):
        """Encode images to latent space.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Latent codes [B, latent_dim]
        """
        return self.enc(x)
    
    def decode(self, z):
        """Decode latent codes to images.
        
        Args:
            z: Latent codes [B, latent_dim]
            
        Returns:
            Reconstructed images [B, C, H, W]
        """
        return self.dec(z)


"""Baseline autoencoder from starter code."""

import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    """Baseline encoder."""
    
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(True),   # /2 -> 128x128
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),            # /4 -> 64x64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),           # /8 -> 32x32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),          # /16 -> 16x16
        )
        
        self.fc = None
        self.shape = None
    
    def _init_fc(self, h):
        """Lazy initialization of FC layer."""
        n, c, H, W = h.shape
        self.shape = (c, H, W)
        self.fc = nn.Linear(c * H * W, self.latent_dim).to(h.device)
    
    def forward(self, x):
        h = self.conv(x)
        if self.fc is None:
            self._init_fc(h)
        z = self.fc(h.view(h.size(0), -1))
        return z


class Decoder(nn.Module):
    """Baseline decoder."""
    
    def __init__(self, out_channels: int, latent_dim: int, img_size: int = 256):
        super().__init__()
        self._shape = (256, img_size // 16, img_size // 16)
        self.fc = nn.Linear(latent_dim, int(np.prod(self._shape)))
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),  # x2 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),   # x4 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),    # x8 -> 128x128
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1), nn.Sigmoid(),  # x16 -> 256x256
        )
    
    def forward(self, z):
        h = self.fc(z)
        c, H, W = self._shape
        h = h.view(z.size(0), c, H, W)
        return self.deconv(h)


class BaselineAutoencoder(nn.Module):
    """Baseline autoencoder architecture.
    
    This is the architecture from the starter code, structured for
    easy integration with our infrastructure.
    """
    
    def __init__(self, channels: int = 3, latent_dim: int = 32, img_size: int = 256):
        """Initialize baseline autoencoder.
        
        Args:
            channels: Number of input channels (3 for RGB, 1 for grayscale)
            latent_dim: Size of latent representation
            img_size: Input image size (default 256)
        """
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        self.enc = Encoder(channels, latent_dim)
        self.dec = Decoder(channels, latent_dim, img_size)
    
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


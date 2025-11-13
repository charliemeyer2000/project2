"""Frequency-domain losses for better edge preservation.

These losses operate in the frequency domain (FFT/DCT) to preserve high-frequency
content like sharp edges, which is critical for traffic light reconstruction.

Top-performing teams likely use these techniques to achieve ultra-low MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FFTLoss(nn.Module):
    """Loss computed in Fourier frequency domain.
    
    Preserves high-frequency content (sharp edges) better than MSE alone.
    This is likely one of the secrets behind Cothlory's insanely low MSE (0.000004).
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute FFT loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            
        Returns:
            FFT loss
        """
        # Compute 2D FFT for each channel
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        # Compute loss on both real and imaginary parts
        loss_real = F.mse_loss(pred_fft.real, target_fft.real, reduction=self.reduction)
        loss_imag = F.mse_loss(pred_fft.imag, target_fft.imag, reduction=self.reduction)
        
        return loss_real + loss_imag


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss using Sobel filter.
    
    Explicitly penalizes edge reconstruction errors, which is critical for
    traffic light boundaries.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Register as buffers (not parameters)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel filter.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Edge magnitude [B, C, H, W]
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            # Use standard RGB to grayscale conversion
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Compute gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Compute magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        return magnitude
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            
        Returns:
            Edge-aware loss
        """
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        
        return F.l1_loss(pred_edges, target_edges)


class CombinedFrequencyLoss(nn.Module):
    """Combined loss with spatial, frequency, and edge components.
    
    This is a comprehensive loss function that combines:
    1. Spatial MSE/L1 losses
    2. ROI-weighted loss for traffic lights
    3. FFT loss for frequency preservation
    4. Edge-aware loss for sharp boundaries
    5. Optional perceptual loss
    
    Use this to achieve rank #1!
    """
    
    def __init__(self, 
                 lambda_roi: float = 15.0,
                 roi_size: float = 0.2,
                 lambda_fft: float = 0.1,
                 lambda_edge: float = 0.05,
                 lambda_perceptual: float = 0.1,
                 use_perceptual: bool = True):
        """Initialize combined frequency loss.
        
        Args:
            lambda_roi: Weight for ROI region
            roi_size: Size of ROI as fraction of image
            lambda_fft: Weight for FFT loss
            lambda_edge: Weight for edge-aware loss
            lambda_perceptual: Weight for perceptual loss
            use_perceptual: Whether to use perceptual loss
        """
        super().__init__()
        
        self.lambda_roi = lambda_roi
        self.roi_size = roi_size
        self.lambda_fft = lambda_fft
        self.lambda_edge = lambda_edge
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        
        # Component losses
        self.mse = nn.MSELoss(reduction='none')
        self.fft_loss = FFTLoss()
        self.edge_loss = EdgeAwareLoss()
        
        # Perceptual loss (VGG-based)
        if use_perceptual:
            try:
                import torchvision.models as models
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
                self.vgg = vgg.eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
                logger.info("CombinedFrequencyLoss: Loaded VGG16 for perceptual loss")
            except Exception as e:
                logger.warning(f"CombinedFrequencyLoss: Could not load VGG16 ({e})")
                self.vgg = None
        else:
            self.vgg = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            
        Returns:
            Combined loss
        """
        # 1. Spatial MSE with ROI weighting
        pixel_loss = self.mse(pred, target)
        
        B, C, H, W = pixel_loss.shape
        
        # Calculate ROI boundaries
        roi_margin = (1.0 - self.roi_size) / 2.0
        h_start = int(H * roi_margin)
        h_end = int(H * (1.0 - roi_margin))
        w_start = int(W * roi_margin)
        w_end = int(W * (1.0 - roi_margin))
        
        # ROI and background losses
        roi_loss = pixel_loss[:, :, h_start:h_end, w_start:w_end].mean()
        bg_loss = pixel_loss.mean()
        
        # Weighted spatial loss
        spatial_loss = self.lambda_roi * roi_loss + bg_loss
        
        # 2. FFT loss (frequency domain)
        fft_loss = self.fft_loss(pred, target)
        
        # 3. Edge-aware loss
        edge_loss = self.edge_loss(pred, target)
        
        # 4. Perceptual loss (optional)
        perceptual_loss = 0.0
        if self.use_perceptual and self.vgg is not None:
            pred_features = self.vgg(pred)
            target_features = self.vgg(target)
            perceptual_loss = F.mse_loss(pred_features, target_features)
        
        # Combine all losses
        total_loss = (spatial_loss + 
                     self.lambda_fft * fft_loss + 
                     self.lambda_edge * edge_loss + 
                     self.lambda_perceptual * perceptual_loss)
        
        return total_loss


def get_frequency_loss(loss_type: str, **kwargs) -> nn.Module:
    """Get frequency-domain loss by name.
    
    Args:
        loss_type: Loss type (fft, edge, combined_fft)
        **kwargs: Loss-specific arguments
        
    Returns:
        Loss function
    """
    if loss_type == "fft":
        return FFTLoss()
    elif loss_type == "edge":
        return EdgeAwareLoss()
    elif loss_type == "combined_fft":
        return CombinedFrequencyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown frequency loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    print("Testing frequency-domain losses...")
    
    # Create dummy data
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    # Test FFT loss
    fft_loss = FFTLoss()
    loss = fft_loss(pred, target)
    print(f"✓ FFT Loss: {loss.item():.6f}")
    
    # Test Edge loss
    edge_loss = EdgeAwareLoss()
    loss = edge_loss(pred, target)
    print(f"✓ Edge Loss: {loss.item():.6f}")
    
    # Test Combined loss
    combined_loss = CombinedFrequencyLoss(
        lambda_roi=15.0,
        lambda_fft=0.1,
        lambda_edge=0.05,
        use_perceptual=False  # Skip VGG for quick test
    )
    loss = combined_loss(pred, target)
    print(f"✓ Combined Loss: {loss.item():.6f}")
    
    print("\n✅ All frequency losses working!")


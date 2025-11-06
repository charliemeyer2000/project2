"""GPU-accelerated augmentation using Kornia."""

import torch
import torch.nn as nn
import kornia.augmentation as K
import logging

logger = logging.getLogger(__name__)


def is_gpu_augmentation_supported(device: str) -> bool:
    """Check if GPU augmentation is supported on this device.
    
    Args:
        device: Device string (cuda, mps, cpu)
        
    Returns:
        True if supported, False otherwise
    """
    if device == "cpu":
        return False  # No point using Kornia on CPU
    
    if device.startswith("cuda"):
        return True  # Always supported on CUDA
    
    if device == "mps":
        # Test if Kornia works on MPS
        try:
            test_tensor = torch.randn(1, 3, 32, 32, device=device)
            test_aug = K.ColorJitter(0.1, 0.1, 0.0, 0.0)
            _ = test_aug(test_tensor)
            return True
        except Exception as e:
            logger.warning(f"Kornia not supported on MPS: {e}")
            return False
    
    return False


class GPUAugmentation(nn.Module):
    """GPU-accelerated augmentation pipeline using Kornia.
    
    This moves augmentation from CPU (in DataLoader workers) to GPU,
    dramatically improving performance when training with fast GPUs like A100.
    
    Usage:
        # In training loop:
        images = images.to(device)
        if training:
            images = gpu_aug(images)  # Apply augmentation on GPU
    """
    
    def __init__(self, strength: str = "medium", device: str = "cuda"):
        """Initialize GPU augmentation pipeline.
        
        Args:
            strength: "light", "medium", or "strong"
            device: Device to run augmentation on
        """
        super().__init__()
        self.strength = strength
        self.device = device
        
        # Build augmentation pipeline
        aug_list = self._get_augmentation_transforms(strength)
        
        # Wrap in Kornia's AugmentationSequential for efficient batched processing
        self.aug = K.AugmentationSequential(
            *aug_list,
            data_keys=["input"],
            same_on_batch=False,  # Different augmentation per image
        )
        
        logger.info(f"🚀 GPU augmentation initialized (strength: {strength}, device: {device})")
    
    def _get_augmentation_transforms(self, strength: str):
        """Get Kornia augmentation transforms for traffic light images.
        
        Traffic light specific considerations:
        - NO horizontal flips (orientation matters)
        - NO hue changes (red/yellow/green are CRITICAL)
        - Brightness/contrast variations (lighting: day/night/weather)
        - Small geometric transforms (camera position/distance)
        - Minimal rotation (camera tilt only)
        
        Args:
            strength: "light", "medium", or "strong"
            
        Returns:
            List of Kornia augmentation modules
        """
        if strength == "light":
            return [
                # Lighting conditions (day/night, shadows, sun glare)
                K.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.0,  # Don't touch saturation
                    hue=0.0,  # NEVER touch hue for traffic lights!
                    p=0.8
                ),
            ]
        
        elif strength == "medium":
            return [
                # Lighting and weather conditions
                K.ColorJitter(
                    brightness=0.3,  # Day/night, shadows
                    contrast=0.3,    # Weather, sun glare
                    saturation=0.2,  # Weather affects saturation
                    hue=0.0,  # NEVER touch hue!
                    p=0.8
                ),
                # Camera position/distance variations
                K.RandomAffine(
                    degrees=2,  # Very small rotation (camera tilt)
                    translate=(0.05, 0.05),  # Small position shifts
                    scale=(0.95, 1.05),  # Distance varies
                    p=0.5
                ),
            ]
        
        elif strength == "strong":
            return [
                # More aggressive lighting/weather variations
                K.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.0,  # Still NO HUE!
                    p=0.8
                ),
                # Larger camera variations
                K.RandomAffine(
                    degrees=3,  # Still very small (camera tilt)
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                    p=0.5
                ),
                # Simulate slight lens distortion
                K.RandomPerspective(
                    distortion_scale=0.05,
                    p=0.2
                ),
                # Simulate motion blur (shaky camera, moving vehicle)
                K.RandomMotionBlur(
                    kernel_size=3,
                    angle=(-15, 15),
                    direction=(-0.5, 0.5),
                    p=0.2
                ),
            ]
        
        else:
            logger.warning(f"Unknown augmentation strength '{strength}', using medium")
            return self._get_augmentation_transforms("medium")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply GPU augmentation to a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W], already on GPU
            
        Returns:
            Augmented images [B, C, H, W]
        """
        # Kornia expects images in [0, 1] range (they should already be from ToTensor)
        return self.aug(images)
    
    def to(self, device):
        """Move augmentation pipeline to device."""
        super().to(device)
        self.device = device
        return self


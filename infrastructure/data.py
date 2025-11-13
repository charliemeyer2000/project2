"""Data loading utilities."""

import random
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import logging

logger = logging.getLogger(__name__)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    """Check if path is an image file."""
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def collect_image_paths(data_root: str) -> List[str]:
    """Collect all image paths from data root.
    
    Args:
        data_root: Root directory containing images
        
    Returns:
        List of image paths as strings
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    image_paths = [str(p.resolve()) for p in data_root.rglob("*") if is_image_file(p)]
    
    if not image_paths:
        raise ValueError(f"No images found in {data_root}")
    
    logger.info(f"Found {len(image_paths)} images in {data_root}")
    
    return image_paths


def split_train_val(image_paths: List[str], train_split: float = 0.9, 
                   seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split image paths into train and validation sets.
    
    Args:
        image_paths: List of image paths
        train_split: Fraction of data for training
        seed: Random seed
        
    Returns:
        Tuple of (train_paths, val_paths)
    """
    # Shuffle with seed
    random.seed(seed)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)
    
    # Split
    n_train = int(len(shuffled) * train_split)
    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:]
    
    logger.info(f"Split: {len(train_paths)} train, {len(val_paths)} val")
    
    return train_paths, val_paths


class AutoencoderDataset(Dataset):
    """Dataset for autoencoder training with optional augmentation."""
    
    def __init__(self, image_paths: List[str], img_size: int = 256, 
                 grayscale: bool = False, augment: bool = False,
                 augmentation_strength: str = "medium"):
        """Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            img_size: Target image size
            grayscale: Convert to grayscale if True
            augment: Apply data augmentation if True
            augmentation_strength: "light", "medium", "strong"
        """
        self.image_paths = image_paths
        self.img_size = img_size
        self.grayscale = grayscale
        self.augment = augment
        
        # Base transform (always applied)
        base_transform = [
            transforms.Resize((img_size, img_size), 
                            interpolation=InterpolationMode.BILINEAR)
        ]
        
        if grayscale:
            base_transform.append(transforms.Grayscale(1))
        
        # Augmentation transforms (only for training)
        if augment:
            aug_transforms = self._get_augmentation_transforms(augmentation_strength)
            base_transform.extend(aug_transforms)
        
        base_transform.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(base_transform)
    
    def _get_augmentation_transforms(self, strength: str) -> List:
        """Get augmentation transforms for traffic light images.
        
        Traffic light specific considerations:
        - NO horizontal flips (orientation matters for driving context)
        - NO hue changes (red/yellow/green colors are CRITICAL)
        - Brightness/contrast variations (lighting conditions: day/night/weather)
        - Small translations/zoom (camera position/distance varies)
        - Minimal rotation (camera tilt only, ±1-2 degrees max)
        - Saturation changes (weather affects color intensity)
        
        Args:
            strength: "light", "medium", or "strong"
            
        Returns:
            List of transforms
        """
        if strength == "light":
            return [
                # Lighting conditions (day/night, shadows, sun glare)
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        
        elif strength == "medium":
            return [
                # Lighting and weather conditions
                transforms.ColorJitter(
                    brightness=0.3,  # Day/night, shadows
                    contrast=0.3,    # Weather, sun glare
                    saturation=0.2   # Weather affects saturation (NO HUE!)
                ),
                # Camera position/distance variations
                transforms.RandomAffine(
                    degrees=2,  # Very small rotation (camera tilt only)
                    translate=(0.05, 0.05),  # Small camera position shifts
                    scale=(0.95, 1.05)  # Distance to traffic light varies
                ),
            ]
        
        elif strength == "strong":
            return [
                # More aggressive lighting/weather variations
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3  # Still NO HUE - critical for traffic lights!
                ),
                # Slightly larger camera variations
                transforms.RandomAffine(
                    degrees=3,  # Still very small (camera tilt)
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1)
                ),
                # Simulate slight lens distortion (very mild)
                transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
            ]
        
        else:
            logger.warning(f"Unknown augmentation strength '{strength}', using medium")
            return self._get_augmentation_transforms("medium")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_tensor = self.transform(img)
            return img_tensor, 0  # Return dummy label for compatibility
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a zero tensor as fallback
            channels = 1 if self.grayscale else 3
            return torch.zeros(channels, self.img_size, self.img_size), 0


def probe_workers() -> int:
    """Probe for safe number of dataloader workers.
    
    Returns:
        Number of workers (0 if multiprocessing fails)
    """
    try:
        _ = DataLoader([0], num_workers=2)
        return 2
    except Exception:
        return 0


def create_dataloaders(data_root: str, img_size: int = 256, 
                      grayscale: bool = False, batch_size: int = 64,
                      train_split: float = 0.9, num_workers: int = 2,
                      pin_memory: bool = True, seed: int = 42,
                      shuffle: bool = True, 
                      augment: bool = False,
                      augmentation_strength: str = "medium",
                      device_hints: dict = None) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        data_root: Root directory containing images
        img_size: Target image size
        grayscale: Convert to grayscale if True
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for splitting
        shuffle: Shuffle training data
        augment: Apply data augmentation to training data
        augmentation_strength: "light", "medium", or "strong"
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Collect images
    image_paths = collect_image_paths(data_root)
    
    # Split
    train_paths, val_paths = split_train_val(image_paths, train_split, seed)
    
    # Create datasets
    # Training: WITH augmentation if enabled
    train_dataset = AutoencoderDataset(
        train_paths, img_size, grayscale, 
        augment=augment,
        augmentation_strength=augmentation_strength
    )
    
    # Validation: NEVER augment (we want consistent metrics)
    val_dataset = AutoencoderDataset(
        val_paths, img_size, grayscale,
        augment=False
    )
    
    if augment:
        logger.info(f"✅ Data augmentation enabled (strength: {augmentation_strength})")
        logger.info(f"   Training samples: {len(train_paths)} → Effective: {len(train_paths)} × augmentation variations")
    else:
        logger.info("ℹ️  No data augmentation (set data.augment=true to enable)")
    
    # Probe workers if auto
    if num_workers < 0:
        num_workers = probe_workers()
        logger.info(f"Auto-detected {num_workers} workers")
    
    # Apply device-specific optimizations
    prefetch_factor = 4  # Default
    if device_hints and 'prefetch_factor' in device_hints:
        prefetch_factor = device_hints['prefetch_factor']
    
    # Create dataloaders with optimizations for fast GPUs
    # persistent_workers=True keeps workers alive between epochs (MUCH faster!)
    # prefetch_factor controls how many batches to load ahead
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, "
               f"{len(val_loader)} val batches")
    
    return train_loader, val_loader


"""Model evaluation metrics and scoring."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_latent_dim(model: nn.Module) -> Optional[int]:
    """Extract latent dimension from model.
    
    The server checks for these attributes in order:
    1. model.latent_dim
    2. model.enc.latent_dim
    3. model.enc.fc.out_features
    
    Args:
        model: Autoencoder model
        
    Returns:
        Latent dimension or None if not found
    """
    # Try direct attribute
    if hasattr(model, 'latent_dim'):
        return model.latent_dim
    
    # Try encoder attribute
    if hasattr(model, 'enc'):
        if hasattr(model.enc, 'latent_dim'):
            return model.enc.latent_dim
        
        # Try encoder FC layer
        if hasattr(model.enc, 'fc'):
            if hasattr(model.enc.fc, 'out_features'):
                return model.enc.fc.out_features
    
    logger.warning("Could not determine latent dimension from model")
    return None


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB.
    
    Args:
        model_path: Path to model file
        
    Returns:
        File size in MB
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


@torch.no_grad()
def calculate_mse(model: nn.Module, dataloader, device: str = "cuda") -> float:
    """Calculate mean squared error on a dataset.
    
    Args:
        model: Autoencoder model
        dataloader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Mean MSE across all samples
    """
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        # Handle both (x,) and (x, y) batch formats
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
            
        x = x.to(device)
        
        # Forward pass
        reconstructed = model(x)
        
        # Calculate MSE
        loss = criterion(reconstructed, x)
        
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def calculate_weighted_score(latent_dim: int, full_mse: float, 
                            roi_mse: float, model_size_mb: float) -> float:
    """Calculate the weighted score as done by the server.
    
    Normalization bounds:
    - Latent Dimension: 8–256
    - Full MSE: 5e-4 – 5e-2
    - ROI-MSE: 2e-3 – 8e-2
    - Model Size: 1.0–23.0 MB
    
    Weights:
    - 40% Latent Dim
    - 35% Full MSE
    - 20% ROI-MSE
    - 5% Model Size
    
    Args:
        latent_dim: Latent dimension
        full_mse: Full image MSE
        roi_mse: ROI MSE
        model_size_mb: Model size in MB
        
    Returns:
        Weighted score (0-1, higher is better)
    """
    # Normalization bounds
    ld_min, ld_max = 8, 256
    mse_min, mse_max = 5e-4, 5e-2
    roi_min, roi_max = 2e-3, 8e-2
    size_min, size_max = 1.0, 23.0
    
    # Normalize (inverse scaling - lower is better)
    def normalize(value, min_val, max_val):
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        return 1 - normalized  # Invert so higher is better
    
    ld_score = normalize(latent_dim, ld_min, ld_max)
    mse_score = normalize(full_mse, mse_min, mse_max)
    roi_score = normalize(roi_mse, roi_min, roi_max)
    size_score = normalize(model_size_mb, size_min, size_max)
    
    # Weighted combination (matches competition formula)
    weighted_score = (
        0.40 * ld_score +
        0.30 * mse_score +  # Changed from 0.35 to match competition
        0.25 * roi_score +  # Changed from 0.20 to match competition
        0.05 * size_score
    )
    
    return weighted_score


def evaluate_model(model: nn.Module, train_loader, val_loader, 
                   model_path: str, device: str = "cuda") -> Dict[str, Any]:
    """Comprehensive model evaluation.
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_path: Path to saved model file
        device: Device to run on
        
    Returns:
        Dict with all evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Basic model info
    latent_dim = get_latent_dim(model)
    num_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model_path)
    
    logger.info(f"Latent dim: {latent_dim}")
    logger.info(f"Parameters: {num_params:,}")
    logger.info(f"Model size: {model_size_mb:.4f} MB")
    
    # Calculate MSE
    train_mse = calculate_mse(model, train_loader, device)
    val_mse = calculate_mse(model, val_loader, device)
    
    logger.info(f"Train MSE: {train_mse:.6f}")
    logger.info(f"Val MSE: {val_mse:.6f}")
    
    # Estimate weighted score (using val_mse as proxy for both full and ROI)
    # Note: We don't have ROI-MSE locally, so this is just an estimate
    estimated_score = calculate_weighted_score(
        latent_dim=latent_dim,
        full_mse=val_mse,
        roi_mse=val_mse * 2.0,  # ROI typically has higher MSE
        model_size_mb=model_size_mb
    )
    
    logger.info(f"Estimated weighted score: {estimated_score:.4f}")
    
    return {
        'latent_dim': latent_dim,
        'num_parameters': num_params,
        'model_size_mb': model_size_mb,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'estimated_weighted_score': estimated_score
    }


"""Model architectures for traffic light autoencoder."""

from .baseline import BaselineAutoencoder
from .efficient import EfficientAutoencoder

# Model registry for easy swapping
MODEL_REGISTRY = {
    'baseline': BaselineAutoencoder,
    'efficient': EfficientAutoencoder,
}


def get_model(architecture: str, **kwargs):
    """Get model by name.
    
    Args:
        architecture: Model architecture name (baseline, efficient)
        **kwargs: Model-specific arguments (channels, latent_dim, img_size, etc.)
    
    Returns:
        Model instance
    
    Example:
        >>> model = get_model('efficient', channels=3, latent_dim=16)
    """
    if architecture not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {available}"
        )
    
    model_cls = MODEL_REGISTRY[architecture]
    return model_cls(**kwargs)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'BaselineAutoencoder',
    'EfficientAutoencoder',
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
]

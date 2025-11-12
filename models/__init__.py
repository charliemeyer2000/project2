"""Model architectures for traffic light autoencoder."""

from .baseline import BaselineAutoencoder
from .efficient import EfficientAutoencoder
from .attention_autoencoder import AttentionAutoencoder

# Model registry for easy swapping
MODEL_REGISTRY = {
    'baseline': BaselineAutoencoder,
    'efficient': EfficientAutoencoder,
    'attention': AttentionAutoencoder,
}


def get_model(architecture: str, **kwargs):
    """Get model by name.
    
    Args:
        architecture: Model architecture name (baseline, efficient, attention)
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
    
    # Filter kwargs to only valid parameters for this model
    import inspect
    sig = inspect.signature(model_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return model_cls(**filtered_kwargs)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'BaselineAutoencoder',
    'EfficientAutoencoder',
    'AttentionAutoencoder',
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
]

"""Model checkpointing utilities for TorchScript."""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def save_torchscript(model: nn.Module, save_path: str, 
                     verify: bool = True) -> bool:
    """Save model as TorchScript.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save TorchScript file
        verify: Whether to verify the saved model can be loaded
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        # Ensure model is in eval mode
        model.eval()
        
        # Convert to TorchScript
        logger.info(f"Converting model to TorchScript...")
        scripted_model = torch.jit.script(model)
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        scripted_model.save(str(save_path))
        
        # Check file size
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info(f"💾 Saved TorchScript model: {save_path} ({file_size_mb:.4f} MB)")
        
        if file_size_mb >= 23.0:
            logger.warning(f"⚠️ Model size {file_size_mb:.2f} MB exceeds 23 MB limit!")
        
        # Verify if requested
        if verify:
            logger.info("Verifying saved model...")
            test_model = torch.jit.load(str(save_path))
            logger.info("✅ Model verification successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save TorchScript model: {e}")
        return False


def load_torchscript(model_path: str, device: str = "cpu") -> Optional[torch.jit.ScriptModule]:
    """Load a TorchScript model.
    
    Args:
        model_path: Path to TorchScript file
        device: Device to load model to
        
    Returns:
        Loaded model or None if failed
    """
    try:
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Loading TorchScript model from {model_path}...")
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Loaded model ({file_size_mb:.4f} MB)")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load TorchScript model: {e}")
        return None


def save_checkpoint(model: nn.Module, optimizer, epoch: int, 
                   loss: float, save_path: str, 
                   additional_info: Optional[dict] = None) -> bool:
    """Save a training checkpoint (not TorchScript).
    
    This saves model state dict, optimizer state, and training info
    for resuming training. This is NOT for submission.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        additional_info: Additional info to save
        
    Returns:
        True if successful
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        logger.debug(f"Saved training checkpoint: {save_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False


def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer=None, device: str = "cpu") -> Optional[dict]:
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to
        
    Returns:
        Checkpoint dict or None if failed
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


"""Training loop utilities."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
from typing import Optional, Callable, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MixedLoss(nn.Module):
    """Mixed MSE + L1 loss."""
    
    def __init__(self, lambda_l1: float = 0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        if self.lambda_l1 <= 0:
            return self.mse(pred, target)
        if self.lambda_l1 >= 1:
            return self.l1(pred, target)
        return (1 - self.lambda_l1) * self.mse(pred, target) + self.lambda_l1 * self.l1(pred, target)


def get_loss_fn(loss_type: str, lambda_l1: float = 0.5) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_type: Loss type (mse, l1, mixed)
        lambda_l1: Weight for L1 in mixed loss
        
    Returns:
        Loss function
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "mixed":
        return MixedLoss(lambda_l1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model: nn.Module, dataloader, optimizer, 
                criterion, device: str = "cuda",
                mixed_precision: bool = False,
                log_interval: int = 20,
                epoch: int = 0) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        mixed_precision: Use automatic mixed precision
        log_interval: Log every N batches
        epoch: Current epoch number
        
    Returns:
        Dict with training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    scaler = GradScaler() if mixed_precision else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Handle both (x,) and (x, y) batch formats
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        
        x = x.to(device)
        
        # Forward pass
        if mixed_precision:
            with autocast():
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
        else:
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        
        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            avg_loss = total_loss / total_samples
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    return {
        'train_loss': avg_loss,
        'total_samples': total_samples
    }


@torch.no_grad()
def validate_epoch(model: nn.Module, dataloader, criterion, 
                  device: str = "cuda",
                  mixed_precision: bool = False) -> Dict[str, float]:
    """Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        mixed_precision: Use automatic mixed precision
        
    Returns:
        Dict with validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0
    
    mse_criterion = nn.MSELoss()
    
    for batch in dataloader:
        # Handle both (x,) and (x, y) batch formats
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        
        x = x.to(device)
        
        # Forward pass
        if mixed_precision:
            with autocast():
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                mse = mse_criterion(reconstructed, x)
        else:
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            mse = mse_criterion(reconstructed, x)
        
        # Update metrics
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')
    
    return {
        'val_loss': avg_loss,
        'val_mse': avg_mse,
        'total_samples': total_samples
    }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-6, mode: str = 'min'):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered (patience={self.patience})")
                self.early_stop = True
                return True
        
        return False


def get_optimizer(model: nn.Module, optimizer_name: str, 
                 lr: float, weight_decay: float = 0.0):
    """Get optimizer by name.
    
    Args:
        model: Model to optimize
        optimizer_name: Optimizer name (adam, adamw, sgd)
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, 
                               momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name: Optional[str], 
                 scheduler_params: dict, num_epochs: int, steps_per_epoch: int):
    """Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Scheduler name (step, cosine, onecycle, None)
        scheduler_params: Scheduler parameters
        num_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        
    Returns:
        Scheduler or None
    """
    if scheduler_name is None or scheduler_name == "none":
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "step":
        step_size = scheduler_params.get('step_size', 10)
        gamma = scheduler_params.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    
    elif scheduler_name == "onecycle":
        max_lr = scheduler_params.get('max_lr', optimizer.param_groups[0]['lr'] * 10)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


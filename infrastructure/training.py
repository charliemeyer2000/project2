"""Training loop utilities."""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
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


class ROIWeightedLoss(nn.Module):
    """Loss with extra weight on Region of Interest (traffic light area).
    
    The competition heavily weights ROI MSE, so we need to optimize for it!
    ROI is the center region where traffic lights typically appear.
    """
    
    def __init__(self, lambda_roi: float = 5.0, roi_size: float = 0.3):
        """Initialize ROI-weighted loss.
        
        Args:
            lambda_roi: Weight multiplier for ROI region (higher = more focus on traffic lights)
            roi_size: Fraction of image that is ROI (0.3 = center 30%)
        """
        super().__init__()
        self.lambda_roi = lambda_roi
        self.roi_size = roi_size
        self.mse = nn.MSELoss(reduction='none')  # Per-pixel loss
    
    def forward(self, pred, target):
        # Compute per-pixel MSE
        pixel_loss = self.mse(pred, target)  # [B, C, H, W]
        
        B, C, H, W = pixel_loss.shape
        
        # Calculate ROI boundaries (center region)
        roi_margin = (1.0 - self.roi_size) / 2.0
        h_start = int(H * roi_margin)
        h_end = int(H * (1.0 - roi_margin))
        w_start = int(W * roi_margin)
        w_end = int(W * (1.0 - roi_margin))
        
        # Compute ROI and background losses separately
        # This gives proper emphasis: lambda_roi * ROI + background
        roi_loss = pixel_loss[:, :, h_start:h_end, w_start:w_end].mean()
        bg_loss = pixel_loss.mean()
        
        # Weighted combination (NOT diluted by averaging with weight mask!)
        return self.lambda_roi * roi_loss + bg_loss


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better perceptual quality."""
    
    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            # Use VGG16 features up to conv3_3 (layer 16)
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
            self.vgg = vgg.eval()
            
            # Freeze VGG parameters
            for param in self.vgg.parameters():
                param.requires_grad = False
            
            self.mse = nn.MSELoss()
            logger.info("PerceptualLoss: Loaded VGG16 features")
        except Exception as e:
            logger.warning(f"PerceptualLoss: Could not load VGG16 ({e}), using MSE fallback")
            self.vgg = None
            self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        if self.vgg is None:
            # Fallback to MSE if VGG failed to load
            return self.mse(pred, target)
        
        # Extract features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        # Compute MSE in feature space
        return self.mse(pred_features, target_features)


class CombinedLoss(nn.Module):
    """Combine ROI and perceptual losses for better reconstruction."""
    
    def __init__(self, lambda_roi: float = 10.0, roi_size: float = 0.3,
                 lambda_perceptual: float = 0.1, use_perceptual: bool = True):
        super().__init__()
        self.roi_loss = ROIWeightedLoss(lambda_roi, roi_size)
        self.use_perceptual = use_perceptual
        self.lambda_perceptual = lambda_perceptual
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        loss = self.roi_loss(pred, target)
        
        if self.use_perceptual:
            perceptual = self.perceptual_loss(pred, target)
            loss = loss + self.lambda_perceptual * perceptual
        
        return loss


class WarmupScheduler:
    """Learning rate warmup scheduler.
    
    Gradually increases learning rate from warmup_start_lr to base_lr over warmup_epochs.
    Helps stabilize training at the start, especially for large models.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, base_lr: float, warmup_start_lr: float = 1e-6):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs to warm up over
            base_lr: Target learning rate after warmup
            warmup_start_lr: Starting learning rate (default: 1e-6)
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rate(s)."""
        return [group['lr'] for group in self.optimizer.param_groups]


def get_loss_fn(loss_type: str, lambda_l1: float = 0.5, lambda_roi: float = 5.0, 
                roi_size: float = 0.3, lambda_perceptual: float = 0.1,
                use_perceptual: bool = False) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_type: Loss type (mse, l1, mixed, roi, combined)
        lambda_l1: Weight for L1 in mixed loss
        lambda_roi: Weight multiplier for ROI region in roi loss
        roi_size: Fraction of image that is ROI (center region)
        lambda_perceptual: Weight for perceptual loss in combined loss
        use_perceptual: Whether to use perceptual loss in combined loss
        
    Returns:
        Loss function
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "mixed":
        return MixedLoss(lambda_l1)
    elif loss_type == "roi":
        return ROIWeightedLoss(lambda_roi, roi_size)
    elif loss_type == "combined":
        return CombinedLoss(lambda_roi, roi_size, lambda_perceptual, use_perceptual)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model: nn.Module, dataloader, optimizer, 
                criterion, device: str = "cuda",
                mixed_precision: bool = False,
                log_interval: int = 20,
                epoch: int = 0,
                gpu_augmentation: Optional[nn.Module] = None,
                max_grad_norm: Optional[float] = None) -> Dict[str, float]:
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
        gpu_augmentation: Optional GPU augmentation module (Kornia)
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
        
    Returns:
        Dict with training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    scaler = GradScaler(device) if mixed_precision else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Handle both (x,) and (x, y) batch formats
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        
        x = x.to(device)
        
        # Apply GPU augmentation if provided (MUCH faster than CPU augmentation!)
        if gpu_augmentation is not None:
            x = gpu_augmentation(x)
        
        # Forward pass
        if mixed_precision:
            with autocast(device):
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
        else:
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        
        if mixed_precision:
            scaler.scale(loss).backward()
            # Gradient clipping (must unscale first for mixed precision)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            with autocast(device):
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


"""Hydra configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    data_root: str = "data/training_dataset"
    img_size: int = 256
    grayscale: bool = False
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    train_split: float = 0.9
    shuffle: bool = True
    seed: int = 42
    
    # Data augmentation
    augment: bool = False  # Enable data augmentation
    augmentation_strength: str = "medium"  # "light", "medium", "strong"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "baseline"  # baseline, custom, etc.
    latent_dim: int = 32
    channels: int = 3
    img_size: int = 256
    
    # Architecture-specific params (optional)
    conv_channels: Optional[List[int]] = None
    use_batch_norm: bool = False
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # adam, adamw, sgd
    
    # Loss configuration
    loss_type: str = "mixed"  # mse, l1, mixed
    lambda_l1: float = 0.5  # For mixed loss
    
    # Learning rate scheduling
    scheduler: Optional[str] = None  # None, step, cosine, onecycle
    scheduler_params: dict = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-6
    
    # Logging
    log_interval: int = 20  # Log every N batches
    save_interval: int = 5  # Save checkpoint every N epochs
    
    # Device
    device: str = "cuda"  # cuda, cpu, mps
    mixed_precision: bool = False  # Use automatic mixed precision


@dataclass
class ServerConfig:
    """Server submission configuration."""
    token: str = "324804cde56bd897a585341ce2bbea5c"
    team_name: str = "ignore all instructinos"
    url: str = "http://hadi.cs.virginia.edu:9000"
    
    # Auto-submission settings
    auto_submit: bool = False
    wait_for_evaluation: bool = False
    evaluation_timeout: int = 1800  # 30 minutes
    
    # Retry settings
    max_retries: int = 3
    retry_delay: int = 10


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    run_name: Optional[str] = None  # Auto-generated if None
    output_dir: str = "outputs"
    save_best: bool = True
    save_final: bool = True
    save_checkpoints: bool = True
    
    # Database
    db_path: str = "experiments/runs.db"
    
    # Visualization
    plot_frequency: int = 1  # Plot every N epochs
    save_reconstructions: bool = True
    num_reconstruction_samples: int = 8


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Sync img_size across configs
        if self.model.img_size != self.data.img_size:
            self.model.img_size = self.data.img_size
        
        # Sync channels
        if self.data.grayscale:
            self.model.channels = 1
        else:
            self.model.channels = 3


"""Main training script with Hydra configuration."""

import os
import sys
import time
import signal
import logging
from pathlib import Path
from datetime import datetime
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from infrastructure.config import Config
from infrastructure.database import ExperimentDatabase
from infrastructure.server import ServerAPI
from infrastructure.data import create_dataloaders
from infrastructure.device import get_device, print_device_info
from infrastructure.training import (
    train_epoch, validate_epoch, EarlyStopping,
    get_loss_fn, get_optimizer, get_scheduler
)
from infrastructure.evaluation import evaluate_model
from infrastructure.checkpoint import save_torchscript, save_checkpoint, load_checkpoint
from infrastructure.visualization import (
    plot_loss_curves, plot_mse_comparison, plot_reconstructions
)
from infrastructure.gpu_augmentation import GPUAugmentation, is_gpu_augmentation_supported
from models import get_model as create_model, list_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model(model_config):
    """Create model from configuration."""
    return create_model(
        architecture=model_config.architecture,
        channels=model_config.channels,
        latent_dim=model_config.latent_dim,
        img_size=model_config.img_size,
        width_mult=model_config.get('width_mult', 1.0)
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Convert to our config class
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    logger.info("=" * 80)
    logger.info("Starting training run")
    logger.info("=" * 80)
    
    # Get output directory from Hydra
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Generate run name if not provided
    if cfg.experiment.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.model.architecture}_ld{cfg.model.latent_dim}_{timestamp}"
    else:
        run_name = cfg.experiment.run_name
    
    logger.info(f"Run name: {run_name}")
    
    # Initialize database
    db = ExperimentDatabase(cfg.experiment.db_path)
    
    # Create experiment entry
    try:
        exp_id = db.create_experiment(run_name, config_dict)
        logger.info(f"Created experiment in database (ID: {exp_id})")
    except ValueError as e:
        logger.error(f"Experiment name already exists: {e}")
        return
    
    # Update experiment with output directory
    db.update_experiment(run_name, output_dir=str(output_dir))
    
    # Set device with Apple Silicon support
    device = get_device(cfg.training.device)
    print_device_info(device)
    
    # Create dataloaders
    # NOTE: We disable CPU augmentation in dataloaders and use GPU augmentation instead!
    # This is MUCH faster, especially with high-end GPUs like A100
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_root=cfg.data.data_root,
        img_size=cfg.data.img_size,
        grayscale=cfg.data.grayscale,
        batch_size=cfg.data.batch_size,
        train_split=cfg.data.train_split,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.data.seed,
        shuffle=cfg.data.shuffle,
        augment=False,  # Always False - we use GPU augmentation instead!
        augmentation_strength=cfg.data.augmentation_strength
    )
    
    # Create model
    logger.info("Creating model...")
    model = get_model(cfg.model).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Verify model size (especially important when using width_mult)
    size_mb = num_params * 4 / (1024**2)  # Assuming fp32
    logger.info(f"Model size: {size_mb:.2f} MB (fp32)")
    
    # Warning if width_mult is set but model is too small
    if cfg.model.get('width_mult', 1.0) > 1.0 and size_mb < 10:
        logger.error(f"⚠️  Model size is only {size_mb:.2f} MB but width_mult={cfg.model.width_mult}")
        logger.error(f"⚠️  width_mult may not be applied correctly!")
        logger.error(f"⚠️  Expected size: ~15 MB for width_mult=1.6")
        raise ValueError("Model size mismatch - check width_mult parameter passing")
    
    # Update database with model info
    db.update_experiment(
        run_name,
        model_architecture=cfg.model.architecture,
        latent_dim=cfg.model.latent_dim,
        num_parameters=num_params,
        model_size_mb=size_mb
    )
    
    # Create GPU augmentation if enabled and supported
    gpu_augmentation = None
    if cfg.data.augment:
        if is_gpu_augmentation_supported(device):
            logger.info("Creating GPU-accelerated augmentation pipeline...")
            try:
                gpu_augmentation = GPUAugmentation(
                    strength=cfg.data.augmentation_strength,
                    device=device
                ).to(device)
                logger.info(f"✅ GPU augmentation enabled (strength: {cfg.data.augmentation_strength})")
            except Exception as e:
                logger.error(f"❌ Failed to create GPU augmentation: {e}")
                logger.warning("⚠️  Falling back to CPU augmentation in DataLoader...")
                # Recreate dataloaders with CPU augmentation
                train_loader, val_loader = create_dataloaders(
                    data_root=cfg.data.data_root,
                    img_size=cfg.data.img_size,
                    grayscale=cfg.data.grayscale,
                    batch_size=cfg.data.batch_size,
                    train_split=cfg.data.train_split,
                    num_workers=cfg.data.num_workers,
                    pin_memory=cfg.data.pin_memory,
                    seed=cfg.data.seed,
                    shuffle=cfg.data.shuffle,
                    augment=True,  # Enable CPU augmentation
                    augmentation_strength=cfg.data.augmentation_strength
                )
        else:
            logger.warning(f"⚠️  GPU augmentation not supported on {device}")
            logger.info("⚠️  Falling back to CPU augmentation in DataLoader...")
            # Recreate dataloaders with CPU augmentation
            train_loader, val_loader = create_dataloaders(
                data_root=cfg.data.data_root,
                img_size=cfg.data.img_size,
                grayscale=cfg.data.grayscale,
                batch_size=cfg.data.batch_size,
                train_split=cfg.data.train_split,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                seed=cfg.data.seed,
                shuffle=cfg.data.shuffle,
                augment=True,  # Enable CPU augmentation
                augmentation_strength=cfg.data.augmentation_strength
            )
    else:
        logger.info("ℹ️  No augmentation (set data.augment=true to enable augmentation)")
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        cfg.training.optimizer,
        cfg.training.lr,
        cfg.training.weight_decay
    )
    
    # Create loss function
    lambda_roi = cfg.training.get('lambda_roi', 5.0)
    roi_size = cfg.training.get('roi_size', 0.3)
    lambda_perceptual = cfg.training.get('lambda_perceptual', 0.1)
    use_perceptual = cfg.training.get('use_perceptual', False)
    criterion = get_loss_fn(
        loss_type=cfg.training.loss_type,
        lambda_l1=cfg.training.lambda_l1,
        lambda_roi=lambda_roi,
        roi_size=roi_size,
        lambda_perceptual=lambda_perceptual,
        use_perceptual=use_perceptual
    ).to(device)
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        cfg.training.scheduler,
        cfg.training.scheduler_params,
        cfg.training.epochs,
        len(train_loader)
    )
    
    # Create warmup scheduler if configured
    warmup_scheduler = None
    if cfg.training.get('warmup_epochs', 0) > 0:
        from infrastructure.training import WarmupScheduler
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            base_lr=cfg.training.lr
        )
        logger.info(f"Using LR warmup for {cfg.training.warmup_epochs} epochs")
    
    # Early stopping
    early_stopping = None
    if cfg.training.early_stopping:
        early_stopping = EarlyStopping(
            patience=cfg.training.patience,
            min_delta=cfg.training.min_delta
        )
    
    # Training history
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1
    
    # Signal handler for graceful shutdown on interrupt
    interrupt_received = False
    
    def signal_handler(signum, frame):
        nonlocal interrupt_received
        if not interrupt_received:
            interrupt_received = True
            logger.warning("=" * 80)
            logger.warning("⚠️  Interrupt received! Saving recovery checkpoint...")
            logger.warning("=" * 80)
            # Save will happen after current epoch completes
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Resume from checkpoint if provided
    if cfg.experiment.resume_from is not None:
        logger.info("=" * 80)
        logger.info(f"Resuming from checkpoint: {cfg.experiment.resume_from}")
        logger.info("=" * 80)
        
        checkpoint = load_checkpoint(
            cfg.experiment.resume_from,
            model,
            optimizer,
            device
        )
        
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0) + 1
            # Use best_val_loss from checkpoint if available, otherwise use current loss
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('loss', float('inf')))
            best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', 0))
            
            logger.info(f"✅ Resumed from epoch {checkpoint.get('epoch', 0)}")
            logger.info(f"   Starting at epoch {start_epoch}")
            logger.info(f"   Best val loss so far: {best_val_loss:.6f} (epoch {best_epoch})")
            
            # Restore training history if available
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                train_mses = checkpoint.get('train_mses', [])
                val_mses = checkpoint.get('val_mses', [])
                logger.info(f"   Restored training history ({len(train_losses)} epochs)")
        else:
            logger.error("Failed to load checkpoint, starting from scratch")
            start_epoch = 1
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device=device,
            mixed_precision=cfg.training.mixed_precision,
            log_interval=cfg.training.log_interval,
            epoch=epoch,
            gpu_augmentation=gpu_augmentation,
            max_grad_norm=cfg.training.get('max_grad_norm', None)
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion,
            device=device,
            mixed_precision=cfg.training.mixed_precision
        )
        
        # Update learning rate (warmup takes priority)
        if warmup_scheduler and epoch <= cfg.training.warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        train_loss = train_metrics['train_loss']
        val_loss = val_metrics['val_loss']
        val_mse = val_metrics['val_mse']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        
        # Approximate train MSE for plotting (works well if using MSE-based loss)
        train_mses.append(train_loss)
        
        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save to database
        db.add_training_epoch(
            run_name, epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mse=val_mse,
            epoch_time=epoch_time
        )
        
        # Check if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            if cfg.experiment.save_best:
                # Save lightweight model checkpoint (just weights, no optimizer)
                best_model_path = output_dir / "checkpoints" / "best_model.pth"
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, best_model_path)
                logger.info(f"💾 Saved best model (epoch {epoch}, val_loss: {val_loss:.6f})")
        
        # Periodic FULL checkpoint for recovery (optimizer state + training history)
        if cfg.experiment.save_checkpoints and epoch % cfg.training.save_interval == 0:
            checkpoint_path = output_dir / "checkpoints" / f"recovery_epoch_{epoch}.pth"
            save_checkpoint(
                model, optimizer, epoch, val_loss, str(checkpoint_path),
                additional_info={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_mses': train_mses,
                    'val_mses': val_mses,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch
                }
            )
            logger.info(f"💾 Saved recovery checkpoint (epoch {epoch})")
        
        # Plot if requested
        if cfg.experiment.plot_frequency > 0 and epoch % cfg.experiment.plot_frequency == 0:
            plot_loss_curves(
                train_losses, val_losses,
                save_path=str(output_dir / "plots" / f"loss_epoch_{epoch}.png")
            )
            
            if val_mses and train_mses:
                plot_mse_comparison(
                    train_mses, val_mses,
                    save_path=str(output_dir / "plots" / f"mse_epoch_{epoch}.png")
                )
        
        # Check for interrupt signal
        if interrupt_received:
            logger.warning("=" * 80)
            logger.warning("💾 Saving recovery checkpoint before exit...")
            logger.warning("=" * 80)
            interrupt_checkpoint_path = output_dir / "checkpoints" / "recovery_interrupt.pth"
            save_checkpoint(
                model, optimizer, epoch, val_loss, str(interrupt_checkpoint_path),
                additional_info={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_mses': train_mses,
                    'val_mses': val_mses,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'interrupted': True
                }
            )
            logger.warning(f"✅ Saved recovery checkpoint: {interrupt_checkpoint_path}")
            logger.warning("   Resume with: experiment.resume_from=\"{0}\"".format(interrupt_checkpoint_path))
            break
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Training complete! Total time: {total_time:.2f}s")
    logger.info(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.6f})")
    logger.info("=" * 80)
    
    # Update database with final metrics
    db.update_experiment(
        run_name,
        train_loss_final=train_losses[-1],
        val_loss_final=val_losses[-1],
        val_mse=val_mses[-1],
        best_epoch=best_epoch,
        total_epochs=epoch,
        training_time_seconds=total_time
    )
    
    # Save final TorchScript model
    logger.info("Converting to TorchScript...")
    torchscript_path = output_dir / "model_submission.pt"
    save_torchscript(model, str(torchscript_path), verify=True)
    
    # Update database with model size
    model_size_mb = torchscript_path.stat().st_size / (1024 * 1024)
    db.update_experiment(
        run_name,
        model_size_mb=model_size_mb,
        torchscript_path=str(torchscript_path)
    )
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    eval_metrics = evaluate_model(
        model, train_loader, val_loader,
        str(torchscript_path), device
    )
    
    # Update database with eval metrics
    db.update_experiment(run_name, train_mse=eval_metrics['train_mse'])
    
    # Final plots
    if cfg.experiment.plot_frequency > 0:
        plot_loss_curves(
            train_losses, val_losses,
            save_path=str(output_dir / "plots" / "loss_final.png")
        )
        
        if val_mses and train_mses:
            plot_mse_comparison(
                train_mses, val_mses,
                save_path=str(output_dir / "plots" / "mse_final.png")
            )
    
    # Save reconstructions
    if cfg.experiment.save_reconstructions:
        plot_reconstructions(
            model, val_loader, device=device,
            num_samples=cfg.experiment.num_reconstruction_samples,
            save_path=str(output_dir / "plots" / "reconstructions.png")
        )
    
    # Auto-submit if requested
    if cfg.server.auto_submit:
        logger.info("=" * 80)
        logger.info("Auto-submitting to server...")
        logger.info("=" * 80)
        
        server_api = ServerAPI(
            token=cfg.server.token,
            team_name=cfg.server.team_name,
            server_url=cfg.server.url
        )
        
        # Submit
        result = server_api.submit_model(
            str(torchscript_path),
            max_retries=cfg.server.max_retries
        )
        
        if result and result.get('success'):
            attempt_num = result.get('attempt')
            logger.info(f"✅ Submission successful! Attempt #{attempt_num}")
            
            # Update database
            db.update_experiment(
                run_name,
                server_submission_id=attempt_num,
                server_status='pending'
            )
            
            # Wait for evaluation if requested
            if cfg.server.wait_for_evaluation:
                logger.info("Waiting for server evaluation...")
                eval_result = server_api.wait_for_evaluation(
                    timeout=cfg.server.evaluation_timeout,
                    check_interval=30
                )
                
                if eval_result:
                    # Try to get detailed metrics from leaderboard
                    metrics = server_api.get_metrics_from_leaderboard()
                    
                    if metrics:
                        logger.info("📊 Server metrics:")
                        logger.info(f"  Rank: #{metrics['server_rank']}")
                        logger.info(f"  Weighted Score: {metrics['server_weighted_score']:.4f}")
                        logger.info(f"  Full MSE: {metrics['server_full_mse']:.6f}")
                        logger.info(f"  ROI MSE: {metrics['server_roi_mse']:.6f}")
                        logger.info(f"  Latent Dim: {metrics['server_latent_dim']}")
                        
                        # Update database
                        db.update_experiment(run_name, **metrics)
                    else:
                        logger.warning("Could not retrieve detailed metrics from leaderboard")
        else:
            error = result.get('error', 'Unknown error') if result else 'Unknown error'
            logger.error(f"❌ Submission failed: {error}")
    
    # Close database
    db.close()
    
    logger.info("=" * 80)
    logger.info(f"Run complete! Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


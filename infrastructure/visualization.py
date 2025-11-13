"""Visualization utilities for training analysis."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_loss_curves(train_losses: List[float], val_losses: List[float],
                    save_path: Optional[str] = None, show: bool = False,
                    async_plotter=None):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save plot
        show: Whether to show plot
        async_plotter: Optional AsyncPlotter instance for non-blocking save
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=15)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use async plotting if available (non-blocking)
        if async_plotter:
            async_plotter.save_async(fig, save_path, dpi=300, bbox_inches='tight', close_fig=True)
            logger.info(f"Saving loss curve (async): {save_path}")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved loss curve: {save_path}")
            plt.close()
    
    if show:
        plt.show()
    elif not save_path:
        plt.close()


def plot_mse_comparison(train_mses: List[float], val_mses: List[float],
                       save_path: Optional[str] = None, show: bool = False,
                       async_plotter=None):
    """Plot training and validation MSE comparison.
    
    Args:
        train_mses: List of training MSEs per epoch
        val_mses: List of validation MSEs per epoch
        save_path: Optional path to save plot
        show: Whether to show plot
        async_plotter: Optional AsyncPlotter instance for non-blocking save
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_mses) + 1)
    
    ax.plot(epochs, train_mses, 'b-', label='Train MSE', linewidth=2)
    ax.plot(epochs, val_mses, 'r-', label='Val MSE', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Training and Validation MSE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for MSE
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use async plotting if available (non-blocking)
        if async_plotter:
            async_plotter.save_async(fig, save_path, dpi=300, bbox_inches='tight', close_fig=True)
            logger.info(f"Saving MSE comparison (async): {save_path}")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved MSE comparison: {save_path}")
            plt.close()
    
    if show:
        plt.show()
    elif not save_path:
        plt.close()


@torch.no_grad()
def plot_reconstructions(model, dataloader, device: str = "cuda",
                        num_samples: int = 8,
                        save_path: Optional[str] = None,
                        show: bool = False,
                        async_plotter=None):
    """Plot original vs reconstructed images.
    
    Args:
        model: Autoencoder model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Optional path to save plot
        show: Whether to show plot
        async_plotter: Optional AsyncPlotter instance for non-blocking save
    """
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    if isinstance(batch, (tuple, list)):
        images = batch[0]
    else:
        images = batch
    
    images = images[:num_samples].to(device)
    reconstructed = model(images)
    
    # Move to CPU
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original
        img = images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed
        recon = reconstructed[i].permute(1, 2, 0).numpy()
        recon = np.clip(recon, 0, 1)  # Ensure valid range
        axes[1, i].imshow(recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use async plotting if available (non-blocking)
        if async_plotter:
            async_plotter.save_async(fig, save_path, dpi=300, bbox_inches='tight', close_fig=True)
            logger.info(f"Saving reconstructions (async): {save_path}")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved reconstructions: {save_path}")
            plt.close()
    
    if show:
        plt.show()
    elif not save_path:
        plt.close()


def plot_experiment_comparison(df: pd.DataFrame, 
                               metrics: List[str] = ['val_mse', 'model_size_mb', 'server_weighted_score'],
                               save_path: Optional[str] = None,
                               show: bool = False):
    """Plot comparison of multiple experiments.
    
    Args:
        df: DataFrame with experiment results
        metrics: List of metrics to plot
        save_path: Optional path to save plot
        show: Whether to show plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Filter out None values
        data = df[df[metric].notna()]
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric.replace('_', ' ').title())
            continue
        
        # Bar plot
        ax.bar(range(len(data)), data[metric])
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['run_name'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved experiment comparison: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_leaderboard_history(df: pd.DataFrame, team_name: str,
                             save_path: Optional[str] = None,
                             show: bool = False):
    """Plot leaderboard position over time.
    
    Args:
        df: DataFrame with leaderboard history
        team_name: Team name to track
        save_path: Optional path to save plot
        show: Whether to show plot
    """
    # Filter for our team
    team_data = df[df['team'] == team_name].copy()
    
    if len(team_data) == 0:
        logger.warning(f"No leaderboard data found for team '{team_name}'")
        return
    
    # Convert timestamp to datetime
    team_data['timestamp'] = pd.to_datetime(team_data['timestamp'])
    team_data = team_data.sort_values('timestamp')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Rank over time
    ax1.plot(team_data['timestamp'], team_data['rank'], 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Rank', fontsize=12)
    ax1.set_title(f'Leaderboard Rank Over Time - {team_name}', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Lower rank is better
    ax1.grid(True, alpha=0.3)
    
    # Score over time
    ax2.plot(team_data['timestamp'], team_data['weighted_score'], 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Weighted Score', fontsize=12)
    ax2.set_title('Weighted Score Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved leaderboard history: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_distribution(df: pd.DataFrame, metric: str,
                            save_path: Optional[str] = None,
                            show: bool = False):
    """Plot distribution of a metric across leaderboard.
    
    Args:
        df: DataFrame with leaderboard data
        metric: Metric to plot
        save_path: Optional path to save plot
        show: Whether to show plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(df[metric], bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Distribution of {metric.replace("_", " ").title()}', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2.boxplot(df[metric], vert=True)
    ax2.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax2.set_title(f'Box Plot of {metric.replace("_", " ").title()}', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metric distribution: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


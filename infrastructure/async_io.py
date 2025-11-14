"""Asynchronous I/O operations for maximizing GPU utilization during training.

This module provides thread-based async I/O for:
- Model checkpoint saving (torch.save)
- Matplotlib figure saving
- Database writes

By offloading these blocking I/O operations to background threads, we keep the GPU
continuously utilized without waiting for disk writes.

References:
- PyTorch Lightning AsyncCheckpointIO pattern
- https://gist.github.com/astrofrog/1453933 (async matplotlib)
"""

import logging
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from copy import deepcopy

logger = logging.getLogger(__name__)


class AsyncCheckpointer:
    """Asynchronous checkpoint saver using ThreadPoolExecutor.
    
    This class offloads torch.save() operations to a background thread, allowing
    training to continue without blocking on disk I/O.
    
    Key Features:
    - Automatically moves tensors to CPU before saving
    - Ensures only one checkpoint is saving at a time (prevents memory pressure)
    - Thread-safe checkpoint operations
    
    Example:
        >>> checkpointer = AsyncCheckpointer()
        >>> checkpoint = {'model': model.state_dict(), 'epoch': 10}
        >>> checkpointer.save_async(checkpoint, 'model.pth')
        >>> # Training continues immediately...
        >>> checkpointer.wait()  # Wait before next checkpoint or exit
    """
    
    def __init__(self, max_workers: int = 1):
        """Initialize async checkpointer.
        
        Args:
            max_workers: Number of worker threads (default 1 to avoid memory pressure)
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="checkpoint")
        self.checkpoint_future: Optional[Future] = None
        self.plot_futures: list[Future] = []
        
    def save_async(self, checkpoint: Dict[str, Any], path: str, 
                   log_message: Optional[str] = None) -> Future:
        """Save checkpoint asynchronously.
        
        This method:
        1. Waits for any previous checkpoint to finish (prevents memory buildup)
        2. Deep copies the checkpoint dict
        3. Moves all tensors to CPU
        4. Submits save operation to thread pool
        5. Returns immediately so training can continue
        
        Args:
            checkpoint: Dictionary containing checkpoint data
            path: File path to save checkpoint
            log_message: Optional message to log when save completes
            
        Returns:
            Future object that can be used to wait for completion
        """
        # Wait for previous checkpoint to finish (critical for memory management)
        if self.checkpoint_future is not None:
            logger.debug("Waiting for previous checkpoint to complete...")
            self.checkpoint_future.result()
        
        # Deep copy to avoid modifications during async save
        checkpoint_copy = self._prepare_checkpoint(checkpoint)
        
        # Submit to thread pool
        self.checkpoint_future = self.executor.submit(
            self._save_checkpoint, checkpoint_copy, path, log_message
        )
        
        return self.checkpoint_future
    
    def _prepare_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare checkpoint for async saving by moving tensors to CPU.
        
        This is critical because:
        1. GPU tensors must not be modified during async save
        2. CPU tensors are safe to pickle in background thread
        3. Reduces GPU memory pressure
        
        Args:
            checkpoint: Original checkpoint dictionary
            
        Returns:
            Checkpoint with all tensors moved to CPU
        """
        cpu_checkpoint = {}
        
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                cpu_checkpoint[key] = value.cpu()
            elif isinstance(value, dict):
                # Recursively handle nested dicts (e.g., state_dict)
                cpu_checkpoint[key] = self._prepare_checkpoint(value)
            elif isinstance(value, list):
                cpu_checkpoint[key] = [
                    v.cpu() if isinstance(v, torch.Tensor) else v 
                    for v in value
                ]
            else:
                cpu_checkpoint[key] = value
        
        return cpu_checkpoint
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any], path: str, 
                        log_message: Optional[str] = None):
        """Internal method to save checkpoint (runs in background thread).
        
        Args:
            checkpoint: Checkpoint data (already prepared with CPU tensors)
            path: File path to save to
            log_message: Optional logging message
        """
        try:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint, path)
            
            # Log completion
            if log_message:
                logger.info(log_message)
            else:
                file_size_mb = Path(path).stat().st_size / (1024 * 1024)
                logger.debug(f"✅ Async checkpoint saved: {path} ({file_size_mb:.2f} MB)")
                
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint {path}: {e}")
    
    def wait(self):
        """Wait for all pending checkpoint operations to complete.
        
        Call this before:
        - Starting a new checkpoint
        - Exiting the program
        - Loading a checkpoint you just saved
        """
        if self.checkpoint_future is not None:
            self.checkpoint_future.result()
            self.checkpoint_future = None
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and wait for pending operations.
        
        Args:
            wait: Whether to wait for pending operations to complete
        """
        if wait:
            self.wait()
        self.executor.shutdown(wait=wait)


class AsyncPlotter:
    """Asynchronous matplotlib figure saver using ThreadPoolExecutor.
    
    This class offloads matplotlib savefig() operations to background threads,
    preventing plot saving from blocking training.
    
    Features:
    - Can handle multiple plots concurrently
    - Thread-safe figure saving
    - Automatic figure cleanup
    
    Example:
        >>> plotter = AsyncPlotter(max_workers=2)
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> plotter.save_async(fig, 'plot.png')
        >>> # Training continues immediately...
        >>> plotter.wait_all()  # Wait before exit
    """
    
    def __init__(self, max_workers: int = 2):
        """Initialize async plotter.
        
        Args:
            max_workers: Number of worker threads (2-4 is reasonable for plots)
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="plotter")
        self.futures: list[Future] = []
    
    def save_async(self, fig: plt.Figure, path: str, 
                   dpi: int = 300, bbox_inches: str = 'tight',
                   close_fig: bool = True) -> Future:
        """Save matplotlib figure asynchronously.
        
        Args:
            fig: Matplotlib figure to save
            path: File path to save figure
            dpi: Resolution for saved figure
            bbox_inches: Bounding box setting ('tight' recommended)
            close_fig: Whether to close figure after saving (default True)
            
        Returns:
            Future object that can be used to wait for completion
        """
        # Submit to thread pool
        future = self.executor.submit(
            self._save_figure, fig, path, dpi, bbox_inches, close_fig
        )
        self.futures.append(future)
        
        # Clean up completed futures to avoid memory buildup
        self.futures = [f for f in self.futures if not f.done()]
        
        return future
    
    def _save_figure(self, fig: plt.Figure, path: str, dpi: int, 
                    bbox_inches: str, close_fig: bool):
        """Internal method to save figure (runs in background thread).
        
        Args:
            fig: Matplotlib figure
            path: File path
            dpi: Resolution
            bbox_inches: Bounding box
            close_fig: Whether to close after saving
        """
        try:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            logger.debug(f"✅ Async plot saved: {path}")
            
            # Clean up
            if close_fig:
                plt.close(fig)
                
        except Exception as e:
            logger.error(f"❌ Failed to save plot {path}: {e}")
            if close_fig:
                try:
                    plt.close(fig)
                except:
                    pass
    
    def wait_all(self):
        """Wait for all pending plot operations to complete."""
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error waiting for plot future: {e}")
        self.futures.clear()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and wait for pending operations.
        
        Args:
            wait: Whether to wait for pending operations to complete
        """
        if wait:
            self.wait_all()
        self.executor.shutdown(wait=wait)


class AsyncDatabaseWriter:
    """Asynchronous database write operations using ThreadPoolExecutor.
    
    Offloads database writes to background threads to avoid blocking training.
    
    Example:
        >>> db_writer = AsyncDatabaseWriter()
        >>> db_writer.execute_async(db.add_training_epoch, run_name, epoch, ...)
        >>> # Training continues...
        >>> db_writer.wait_all()
    """
    
    def __init__(self, max_workers: int = 1):
        """Initialize async database writer.
        
        Args:
            max_workers: Number of worker threads (1 is usually sufficient for DB)
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="db_writer")
        self.futures: list[Future] = []
    
    def execute_async(self, func: Callable, *args, **kwargs) -> Future:
        """Execute a database operation asynchronously.
        
        Args:
            func: Database function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Future object
        """
        future = self.executor.submit(func, *args, **kwargs)
        self.futures.append(future)
        
        # Clean up completed futures
        self.futures = [f for f in self.futures if not f.done()]
        
        return future
    
    def wait_all(self):
        """Wait for all pending database operations to complete."""
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in async database operation: {e}")
        self.futures.clear()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and wait for pending operations.
        
        Args:
            wait: Whether to wait for pending operations to complete
        """
        if wait:
            self.wait_all()
        self.executor.shutdown(wait=wait)




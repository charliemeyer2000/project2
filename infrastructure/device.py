"""Device management with optimizations for CUDA and Apple Silicon."""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def optimize_for_cuda():
    """Apply CUDA-specific optimizations.
    
    Enables TensorFloat-32, cuDNN benchmarking, and other performance features.
    """
    logger.info("🚀 Applying CUDA optimizations...")
    
    # Enable TF32 for Ampere+ GPUs (3x speedup with minimal accuracy loss)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        
        # Enable cuDNN deterministic mode if needed (disable for max speed)
        # torch.backends.cudnn.deterministic = True
        
        logger.info("  ✅ TensorFloat-32 enabled")
        logger.info("  ✅ cuDNN benchmark enabled")
        logger.info(f"  ✅ Using {torch.cuda.device_count()} GPU(s)")


def optimize_for_apple_silicon():
    """Apply Apple Silicon (MPS) optimizations.
    
    Sets environment variables and thread counts for optimal MPS performance.
    Returns recommended dataloader settings.
    """
    logger.info("🍎 Applying Apple Silicon optimizations...")
    
    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['MKL_NUM_THREADS'] = '10'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '10'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    
    # Set PyTorch threads
    torch.set_num_threads(10)
    
    logger.info(f"  ✅ PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"  ✅ Recommended: batch_size=128-192, num_workers=8, pin_memory=False")
    
    return {
        'num_workers_multiplier': 2.0,  # Increase num_workers by 2x for MPS
        'batch_size_multiplier': 2.0,   # Increase batch_size by 2x for MPS
        'pin_memory': False,             # Disable pin_memory for unified memory
        'prefetch_factor': 10            # Higher prefetch for MPS
    }


def optimize_for_cpu():
    """Apply CPU-specific optimizations.
    
    Sets thread counts for optimal CPU performance.
    """
    logger.info("💻 Applying CPU optimizations...")
    
    # Set optimal thread count for CPU
    num_threads = min(torch.get_num_threads(), 16)  # Cap at 16 for efficiency
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    logger.info(f"  ✅ PyTorch threads: {num_threads}")


def get_device(preferred_device: str = "auto") -> tuple:
    """Get the best available device with automatic optimization.
    
    Priority (when auto): CUDA > MPS > CPU
    
    Args:
        preferred_device: Preferred device ("auto", "cuda", "mps", "cpu")
        
    Returns:
        Tuple of (device_string, optimization_hints_dict)
    """
    opt_hints = {}
    
    if preferred_device == "auto":
        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = "cuda"
            optimize_for_cuda()
            logger.info("✅ Using CUDA (NVIDIA GPU)")
        elif torch.backends.mps.is_available():
            device = "mps"
            opt_hints = optimize_for_apple_silicon()
            logger.info("✅ Using MPS (Apple Silicon GPU)")
        else:
            device = "cpu"
            optimize_for_cpu()
            logger.info("⚠️  Using CPU (no GPU available)")
    
    elif preferred_device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            optimize_for_cuda()
            logger.info("✅ Using CUDA (NVIDIA GPU)")
        else:
            logger.warning("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"
            optimize_for_cpu()
    
    elif preferred_device == "mps":
        if torch.backends.mps.is_available():
            device = "mps"
            opt_hints = optimize_for_apple_silicon()
            logger.info("✅ Using MPS (Apple Silicon GPU)")
        else:
            logger.warning("⚠️  MPS not available, falling back to CPU")
            device = "cpu"
            optimize_for_cpu()
    
    elif preferred_device == "cpu":
        device = "cpu"
        optimize_for_cpu()
        logger.info("💻 Using CPU (forced)")
    
    else:
        logger.warning(f"Unknown device '{preferred_device}', using auto-detect")
        return get_device("auto")
    
    return device, opt_hints


def print_device_info(device: str):
    """Print detailed device information.
    
    Args:
        device: Device string
    """
    logger.info("=" * 60)
    logger.info("Device Information")
    logger.info("=" * 60)
    logger.info(f"Device: {device.upper()}")
    
    if device == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")
        logger.info(f"Compute Capability: {props.major}.{props.minor}")
        logger.info(f"Multi-Processor Count: {props.multi_processor_count}")
        logger.info(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        logger.info(f"TF32 Matmul: {torch.backends.cuda.matmul.allow_tf32}")
    
    elif device == "mps":
        logger.info("Platform: Apple Silicon")
        logger.info(f"MPS Available: {torch.backends.mps.is_available()}")
        logger.info(f"PyTorch Threads: {torch.get_num_threads()}")
        logger.info("Optimizations: Memory management, fallback enabled")
    
    elif device == "cpu":
        logger.info(f"CPU Threads: {torch.get_num_threads()}")
        import multiprocessing
        logger.info(f"Physical Cores: {multiprocessing.cpu_count()}")
    
    logger.info("=" * 60)


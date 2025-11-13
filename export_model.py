#!/usr/bin/env python3
"""Export PyTorch checkpoint to TorchScript for submission."""

import argparse
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

from models import get_model
from infrastructure.checkpoint import save_torchscript
from infrastructure.server import ServerAPI
from infrastructure.database import ExperimentDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_architecture_from_state_dict(state_dict: dict) -> str:
    """Infer model architecture from state_dict keys."""
    keys = list(state_dict.keys())
    
    # Check for attention-specific keys
    if any('fc.0.weight' in k or 'fc.2.weight' in k for k in keys):
        return 'attention'
    
    # Check for depthwise separable convolutions (efficient model)
    if any('depthwise' in k for k in keys):
        return 'efficient'
    
    # Default to baseline
    return 'baseline'


def infer_latent_dim_from_state_dict(state_dict: dict) -> int:
    """Infer latent dimension from state_dict."""
    # Look for the encoder's final linear layer (fc) output dimension
    if 'enc.fc.weight' in state_dict:
        return state_dict['enc.fc.weight'].shape[0]
    
    # Fallback: look for decoder's first linear layer input dimension
    if 'dec.fc.weight' in state_dict:
        return state_dict['dec.fc.weight'].shape[1]
    
    # If not found, return default
    return 32


def infer_width_mult_from_state_dict(state_dict: dict, architecture: str) -> float:
    """Infer width_mult from state_dict by checking actual channel dimensions.
    
    Only applicable to 'attention' architecture which supports width_mult.
    """
    if architecture != 'attention':
        return 1.0
    
    # Check c2 (second conv layer) - more reliable due to int truncation
    if 'enc.conv2.0.weight' in state_dict:
        actual_c2 = state_dict['enc.conv2.0.weight'].shape[0]
        base_c2 = 64
        width_mult = actual_c2 / base_c2
        return width_mult
    
    # Fallback to c1
    if 'enc.conv1.0.weight' in state_dict:
        actual_c1 = state_dict['enc.conv1.0.weight'].shape[0]
        base_c1 = 32
        width_mult = actual_c1 / base_c1
        return width_mult
    
    # Fallback
    return 1.0


def infer_model_params_from_state_dict(state_dict: dict, config: dict, architecture: str) -> dict:
    """Infer all model parameters from state_dict and config.
    
    Args:
        state_dict: Model state dictionary
        config: Configuration from checkpoint (may be incorrect)
        architecture: Model architecture ('baseline', 'efficient', 'attention')
        
    Returns:
        Dictionary of model parameters to pass to get_model()
    """
    model_config = config.get('model', {})
    
    params = {
        'latent_dim': infer_latent_dim_from_state_dict(state_dict),
        'channels': model_config.get('channels', 3),
        'img_size': model_config.get('img_size', 256),
    }
    
    # Architecture-specific parameters
    if architecture == 'attention':
        # Infer width_mult from actual weights (config might be wrong!)
        params['width_mult'] = infer_width_mult_from_state_dict(state_dict, architecture)
        
        # These should be in config (they don't affect weight shapes)
        params['use_skip_connections'] = model_config.get('use_skip_connections', True)
        params['activation_type'] = model_config.get('activation_type', 'tanh')
        params['norm_type'] = model_config.get('norm_type', 'group')
    elif architecture == 'efficient':
        params['base_channels'] = model_config.get('base_channels', 32)
    
    return params


def export_checkpoint(checkpoint_path: str, output_path: str = None, 
                     submit: bool = False, wait: bool = False,
                     run_name: str = None) -> str:
    """Export checkpoint to TorchScript.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Output path for .pt file (default: same dir as checkpoint)
        submit: Whether to submit to server
        wait: Whether to wait for evaluation
        run_name: Run name for database tracking
        
    Returns:
        Path to exported model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Determine output path
    if output_path is None:
        output_path = checkpoint_path.parent / "model_submission.pt"
    else:
        output_path = Path(output_path)
    
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Extract model info from config (might be wrong!)
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    config_architecture = model_config.get('architecture', 'baseline')
    config_latent_dim = model_config.get('latent_dim', 32)
    
    # Infer actual architecture from state_dict
    actual_architecture = infer_architecture_from_state_dict(state_dict)
    
    # Infer ALL model parameters from state_dict and config
    model_params = infer_model_params_from_state_dict(state_dict, config, actual_architecture)
    
    # Warn if architecture mismatch
    if config_architecture != actual_architecture:
        logger.warning(f"⚠️  Config says '{config_architecture}' but state_dict indicates '{actual_architecture}'")
        logger.warning(f"   Using: {actual_architecture}")
    
    # Warn if latent_dim mismatch
    if config_latent_dim != model_params['latent_dim']:
        logger.warning(f"⚠️  Config says LD={config_latent_dim} but state_dict indicates LD={model_params['latent_dim']}")
        logger.warning(f"   Using: {model_params['latent_dim']}")
    
    # Warn if width_mult was inferred differently (attention only)
    if actual_architecture == 'attention':
        config_width_mult = model_config.get('width_mult', 1.0)
        if abs(config_width_mult - model_params['width_mult']) > 0.01:
            logger.warning(f"⚠️  Config says width_mult={config_width_mult} but state_dict indicates {model_params['width_mult']:.2f}")
            logger.warning(f"   Using: {model_params['width_mult']:.2f}")
    
    architecture = actual_architecture
    
    logger.info(f"   Architecture: {architecture}")
    logger.info(f"   Latent dim: {model_params['latent_dim']}")
    if architecture == 'attention':
        logger.info(f"   Width mult: {model_params['width_mult']:.2f}")
        logger.info(f"   Skip connections: {model_params['use_skip_connections']}")
        logger.info(f"   Activation: {model_params['activation_type']}")
        logger.info(f"   Normalization: {model_params['norm_type']}")
    logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"   Val MSE: {checkpoint.get('val_mse', 'unknown')}")
    
    # Create model with all inferred parameters
    logger.info(f"\n🔨 Creating model...")
    model = get_model(architecture, **model_params)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to TorchScript
    logger.info(f"📦 Exporting to TorchScript...")
    save_torchscript(model, str(output_path))
    
    # Verify TorchScript model can be loaded and latent_dim is accessible
    logger.info(f"🔍 Verifying TorchScript export...")
    try:
        loaded_model = torch.jit.load(str(output_path))
        
        # Verify latent_dim is accessible
        if hasattr(loaded_model, 'latent_dim'):
            loaded_latent_dim = loaded_model.latent_dim
            if loaded_latent_dim == model_params['latent_dim']:
                logger.info(f"   ✅ latent_dim accessible: {loaded_latent_dim}")
            else:
                logger.error(f"   ❌ latent_dim mismatch! Expected {model_params['latent_dim']}, got {loaded_latent_dim}")
                raise ValueError("TorchScript latent_dim mismatch")
        else:
            logger.error(f"   ❌ latent_dim NOT accessible in TorchScript model!")
            logger.error(f"   This will cause 'could not infer latent_dim' error on server!")
            raise ValueError("TorchScript model missing latent_dim attribute")
        
        # Verify encoder also exposes latent_dim (some servers check this)
        if hasattr(loaded_model, 'enc') and hasattr(loaded_model.enc, 'latent_dim'):
            logger.info(f"   ✅ enc.latent_dim accessible: {loaded_model.enc.latent_dim}")
        else:
            logger.warning(f"   ⚠️  enc.latent_dim not accessible (may cause issues on some servers)")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = loaded_model(dummy_input)
            if output.shape == (1, 3, 256, 256):
                logger.info(f"   ✅ Inference test passed: {dummy_input.shape} -> {output.shape}")
            else:
                logger.error(f"   ❌ Inference shape mismatch! Expected (1,3,256,256), got {output.shape}")
                raise ValueError("TorchScript inference failed")
                
    except Exception as e:
        logger.error(f"❌ TorchScript verification failed: {e}")
        logger.error(f"   Model may not work on submission server!")
        raise
    
    # Check size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Export complete and verified!")
    logger.info(f"   Path: {output_path}")
    logger.info(f"   Size: {size_mb:.2f} MB")
    
    # Submit if requested
    if submit:
        logger.info(f"\n🚀 Submitting to server...")
        
        # Load server config from main config
        config_path = Path(__file__).parent / "configs" / "config.yaml"
        cfg = OmegaConf.load(config_path)
        
        api = ServerAPI(
            token=cfg.server.token,
            team_name=cfg.server.team_name,
            server_url=cfg.server.url
        )
        
        success = api.submit_model(str(output_path))
        
        if success:
            logger.info(f"✅ Submitted successfully!")
            
            if wait:
                logger.info(f"\n⏳ Waiting for evaluation...")
                metrics = api.wait_for_evaluation(timeout=1800)
                
                if metrics:
                    logger.info(f"\n📊 Results:")
                    logger.info(f"   Weighted Score: {metrics.get('weighted_score', 'N/A')}")
                    logger.info(f"   Full MSE: {metrics.get('full_mse', 'N/A')}")
                    logger.info(f"   ROI MSE: {metrics.get('roi_mse', 'N/A')}")
                    logger.info(f"   Model Size: {metrics.get('model_size_mb', 'N/A')} MB")
                    
                    # Update database if run_name provided
                    if run_name:
                        try:
                            db = ExperimentDatabase()
                            db.update_experiment(
                                run_name,
                                server_weighted_score=metrics.get('weighted_score'),
                                server_full_mse=metrics.get('full_mse'),
                                server_roi_mse=metrics.get('roi_mse'),
                                server_latent_dim=metrics.get('latent_dim'),
                                server_model_size_mb=metrics.get('model_size_mb'),
                                server_status='successful'
                            )
                            db.close()
                            logger.info(f"✅ Updated database for run: {run_name}")
                        except Exception as e:
                            logger.warning(f"⚠️  Could not update database: {e}")
                else:
                    logger.warning(f"⚠️  Evaluation timed out or failed")
        else:
            logger.error(f"❌ Submission failed")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch checkpoint to TorchScript for submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export checkpoint to TorchScript
  python export_model.py outputs/2025-11-05/12-34-56/best_checkpoint.pth
  
  # Export and submit
  python export_model.py outputs/2025-11-05/12-34-56/best_checkpoint.pth --submit
  
  # Export, submit, and wait for results
  python export_model.py outputs/2025-11-05/12-34-56/best_checkpoint.pth --submit --wait
  
  # Specify output path
  python export_model.py best_checkpoint.pth -o my_model.pt
  
  # Export and submit with database tracking
  python export_model.py best_checkpoint.pth --submit --wait --run-name attention_ld16_local_test
"""
    )
    
    parser.add_argument(
        "checkpoint",
        help="Path to checkpoint file (.pth)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for TorchScript model (default: same dir as checkpoint)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit to server after export"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for evaluation after submission"
    )
    parser.add_argument(
        "--run-name",
        help="Run name for database tracking (used with --submit --wait)"
    )
    
    args = parser.parse_args()
    
    export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        submit=args.submit,
        wait=args.wait,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()




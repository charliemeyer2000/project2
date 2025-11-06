#!/usr/bin/env python3
"""Export PyTorch checkpoint to TorchScript for submission."""

import argparse
import sys
from pathlib import Path
import torch

from models import get_model
from infrastructure.checkpoint import save_torchscript
from infrastructure.server import ServerAPI
from infrastructure.database import ExperimentDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # Extract model info
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'baseline')
    latent_dim = model_config.get('latent_dim', 32)
    
    logger.info(f"   Architecture: {architecture}")
    logger.info(f"   Latent dim: {latent_dim}")
    logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"   Val MSE: {checkpoint.get('val_mse', 'unknown')}")
    
    # Create model
    logger.info(f"\n🔨 Creating model...")
    model = get_model(architecture, latent_dim=latent_dim)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to TorchScript
    logger.info(f"📦 Exporting to TorchScript...")
    save_torchscript(model, str(output_path))
    
    # Check size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Exported: {output_path}")
    logger.info(f"   Size: {size_mb:.2f} MB")
    
    # Submit if requested
    if submit:
        logger.info(f"\n🚀 Submitting to server...")
        api = ServerAPI()
        
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


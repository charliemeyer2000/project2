"""Grid search runner for hyperparameter sweeps."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml
import itertools
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sweep_config(config_path: str) -> Dict[str, Any]:
    """Load sweep configuration from YAML file.
    
    Args:
        config_path: Path to sweep config file
        
    Returns:
        Sweep configuration dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_sweep_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for grid search.
    
    Args:
        sweep_config: Sweep configuration with parameter grids
        
    Returns:
        List of parameter combinations
    """
    # Extract parameter grids
    param_grids = sweep_config.get('parameters', {})
    
    # Generate all combinations
    param_names = list(param_grids.keys())
    param_values = [param_grids[name] for name in param_names]
    
    combinations = []
    for values in itertools.product(*param_values):
        combination = dict(zip(param_names, values))
        combinations.append(combination)
    
    logger.info(f"Generated {len(combinations)} parameter combinations")
    return combinations


def run_training(params: Dict[str, Any], base_args: List[str]) -> bool:
    """Run training with specific parameters.
    
    Args:
        params: Parameter overrides
        base_args: Base arguments for training script
        
    Returns:
        True if successful, False otherwise
    """
    # Build command
    cmd = ["python", "train.py"] + base_args
    
    # Add parameter overrides
    for key, value in params.items():
        cmd.append(f"{key}={value}")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Error running training: {e}")
        return False


def main():
    """Main sweep entry point."""
    parser = argparse.ArgumentParser(
        description="Grid search runner for hyperparameter sweeps"
    )
    parser.add_argument("config", help="Path to sweep configuration YAML file")
    parser.add_argument("--parallel", action="store_true",
                       help="Run sweeps in parallel (not implemented yet)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without running")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue sweep even if a run fails")
    
    args = parser.parse_args()
    
    # Load sweep config
    logger.info(f"Loading sweep config from: {args.config}")
    sweep_config = load_sweep_config(args.config)
    
    # Generate combinations
    combinations = generate_sweep_combinations(sweep_config)
    
    # Get base arguments
    base_args = sweep_config.get('base_args', [])
    
    logger.info("=" * 80)
    logger.info(f"Starting sweep with {len(combinations)} runs")
    logger.info("=" * 80)
    
    # Run sweep
    successful = 0
    failed = 0
    
    for i, params in enumerate(combinations, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Run {i}/{len(combinations)}")
        logger.info(f"Parameters: {params}")
        logger.info("=" * 80)
        
        if args.dry_run:
            logger.info("DRY RUN - Not executing")
            continue
        
        success = run_training(params, base_args)
        
        if success:
            successful += 1
            logger.info(f"✅ Run {i} completed successfully")
        else:
            failed += 1
            logger.error(f"❌ Run {i} failed")
            
            if not args.continue_on_error:
                logger.error("Stopping sweep due to failure")
                break
        
        # Small delay between runs
        if i < len(combinations):
            time.sleep(2)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Sweep Complete")
    logger.info("=" * 80)
    logger.info(f"Total runs: {len(combinations)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


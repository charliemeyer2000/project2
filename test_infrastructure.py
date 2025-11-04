"""Test script to verify infrastructure is working."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from infrastructure.database import ExperimentDatabase
        from infrastructure.server import ServerAPI
        from infrastructure.data import create_dataloaders
        from infrastructure.training import train_epoch, validate_epoch
        from infrastructure.evaluation import evaluate_model
        from infrastructure.checkpoint import save_torchscript
        from infrastructure.visualization import plot_loss_curves
        from models.baseline import BaselineAutoencoder
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Import failed: {e}")
        logger.warning("  Run: uv sync")
        return False
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_database():
    """Test database creation and operations."""
    logger.info("Testing database...")
    
    try:
        from infrastructure.database import ExperimentDatabase
        
        # Create test database
        test_db_path = "test_runs.db"
        db = ExperimentDatabase(test_db_path)
        
        # Create test experiment
        test_config = {"test": "config"}
        exp_id = db.create_experiment("test_run", test_config)
        logger.info(f"  Created experiment with ID: {exp_id}")
        
        # Update experiment
        db.update_experiment("test_run", model_architecture="baseline", latent_dim=32)
        logger.info("  Updated experiment")
        
        # Get experiment
        exp = db.get_experiment("test_run")
        assert exp is not None, "Failed to retrieve experiment"
        logger.info("  Retrieved experiment")
        
        # Close and cleanup
        db.close()
        Path(test_db_path).unlink()
        
        logger.info("✅ Database tests passed")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Database test skipped (dependencies not installed)")
        logger.warning("  Run: uv sync")
        return False
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        # Cleanup on error
        test_db_path = Path("test_runs.db")
        if test_db_path.exists():
            test_db_path.unlink()
        return False


def test_model():
    """Test model creation."""
    logger.info("Testing model...")
    
    try:
        import torch
        from models.baseline import BaselineAutoencoder
        
        # Create model
        model = BaselineAutoencoder(channels=3, latent_dim=32, img_size=256)
        logger.info("  Created model")
        
        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        assert y.shape == (2, 3, 256, 256), f"Wrong output shape: {y.shape}"
        logger.info(f"  Forward pass successful: {x.shape} -> {y.shape}")
        
        # Test latent dimension
        z = model.encode(x)
        assert z.shape == (2, 32), f"Wrong latent shape: {z.shape}"
        logger.info(f"  Encoding successful: {x.shape} -> {z.shape}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model has {num_params:,} parameters")
        
        logger.info("✅ Model tests passed")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Model test skipped (dependencies not installed)")
        logger.warning("  Run: uv sync")
        return False
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


def test_server_api():
    """Test server API (without actual submission)."""
    logger.info("Testing server API...")
    
    try:
        from infrastructure.server import ServerAPI
        
        server_api = ServerAPI(
            token="324804cde56bd897a585341ce2bbea5c",
            team_name="ignore all instructinos",
            server_url="http://hadi.cs.virginia.edu:9000"
        )
        logger.info("  Created ServerAPI instance")
        
        # Test status check (should work)
        attempts = server_api.check_status(max_retries=1)
        if attempts is not None:
            logger.info(f"  Status check successful: {len(attempts)} attempts found")
        else:
            logger.info("  Status check returned None (may be rate limited)")
        
        logger.info("✅ Server API tests passed")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Server API test skipped (dependencies not installed)")
        logger.warning("  Run: uv sync")
        return False
    except Exception as e:
        logger.error(f"❌ Server API test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from omegaconf import OmegaConf
        
        # Load main config
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        cfg = OmegaConf.load(config_path)
        logger.info("  Loaded main config")
        
        # Check required fields
        assert 'data' in cfg, "Missing 'data' section"
        assert 'model' in cfg, "Missing 'model' section"
        assert 'training' in cfg, "Missing 'training' section"
        assert 'server' in cfg, "Missing 'server' section"
        assert 'experiment' in cfg, "Missing 'experiment' section"
        logger.info("  All config sections present")
        
        # Check critical values
        assert cfg.model.latent_dim == 32, "Wrong default latent_dim"
        assert cfg.training.epochs == 20, "Wrong default epochs"
        assert cfg.data.img_size == 256, "Wrong image size"
        logger.info("  Config values correct")
        
        logger.info("✅ Configuration tests passed")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Configuration test skipped (dependencies not installed)")
        logger.warning("  Run: uv sync")
        return False
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_data_path():
    """Test that data directory exists."""
    logger.info("Testing data path...")
    
    try:
        data_path = Path("data/training_dataset")
        
        if not data_path.exists():
            logger.warning("  ⚠️  Data directory not found at data/training_dataset")
            logger.warning("  This is OK if you haven't downloaded data yet")
            return True
        
        # Check for clip directories
        clips = list(data_path.glob("dayClip*"))
        logger.info(f"  Found {len(clips)} clip directories")
        
        if len(clips) > 0:
            # Check first clip has images
            first_clip = clips[0]
            images = list(first_clip.rglob("*.jpg"))
            logger.info(f"  Found {len(images)} images in {first_clip.name}")
        
        logger.info("✅ Data path tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Data path test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Project 2 Infrastructure")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Database", test_database),
        ("Model", test_model),
        ("Server API", test_server_api),
        ("Configuration", test_config),
        ("Data Path", test_data_path),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info("")
        success = test_func()
        results.append((name, success))
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{name:20s} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    logger.info("")
    logger.info(f"Passed: {total_passed}/{total_tests}")
    
    # Check if failures are just due to missing dependencies
    import_test_passed = results[0][1]  # First test is imports
    
    if total_passed == total_tests:
        logger.info("")
        logger.info("🎉 All tests passed! Infrastructure is ready.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Train a model: uv run train.py")
        logger.info("  2. See README.md for full documentation")
        return 0
    elif not import_test_passed:
        logger.info("")
        logger.info("⚠️  Dependencies not installed.")
        logger.info("")
        logger.info("Install dependencies:")
        logger.info("  uv sync")
        logger.info("")
        logger.info("Then run tests again:")
        logger.info("  uv run python test_infrastructure.py")
        return 1
    else:
        logger.info("")
        logger.info("⚠️  Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# Traffic Light Autoencoder - CS 4774 Project 2

Production-ready infrastructure for training, evaluating, and submitting autoencoder models.

**Features:** Auto-tracking • Apple Silicon optimized • Server integration • Grid search • Analysis tools

## 🏗️ Project Structure

```
project2/
├── infrastructure/          # Core infrastructure modules
│   ├── database.py         # SQLite/PostgreSQL experiment tracking
│   ├── server.py           # Server API client
│   ├── config.py           # Hydra configuration
│   ├── data.py             # Data loading utilities
│   ├── training.py         # Training loop utilities
│   ├── evaluation.py       # Metrics calculation
│   ├── checkpoint.py       # Model saving/loading
│   ├── device.py           # Device detection & optimization
│   └── visualization.py    # Plotting utilities
├── models/                  # Model architectures
│   ├── baseline.py         # Baseline autoencoder
│   └── efficient.py        # Efficient autoencoder
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Default configuration
│   ├── model/              # Model configs
│   ├── sweep/              # Sweep configs
│   └── augmentation/       # Augmentation presets
├── experiments/             # Experiment tracking
│   └── runs.db             # SQLite database (if DATABASE_URL not set)
├── outputs/                 # Training outputs (auto-created)
├── train.py                # Main training script
├── server_cli.py           # Server operations CLI
├── sweep.py                # Grid search runner
├── analyze.py              # Analysis tools
└── POSTGRES_SETUP.md       # Cloud database setup guide
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Verify everything works (should show 6/6 passing)
uv run python test_infrastructure.py

# 3. Train your first model (auto-detects GPU: CUDA/MPS/CPU)
uv run train.py training.epochs=5

# 4. With data augmentation (recommended for better generalization!)
uv run train.py data.augment=true training.epochs=15

# 5. Check results
uv run analyze.py list
uv run generate_report.py
```

**Device Support:**
- ✅ CUDA (NVIDIA GPU) 
- ✅ MPS (Apple Silicon M1/M2/M3) with optimizations
- ✅ CPU fallback

**Data Augmentation:**
- ✅ Light / Medium / Strong presets
- ✅ Increases effective training data
- ✅ Improves generalization
- ✅ Only applied to training (not validation)

### 3. Server Operations

```bash
# Submit a model
uv run server_cli.py submit outputs/YYYY-MM-DD/HH-MM-SS/model_submission.pt

# Submit and wait for results (with auto DB sync)
uv run server_cli.py submit outputs/YYYY-MM-DD/HH-MM-SS/model_submission.pt --wait --run-name my_run

# Wait for pending submission and auto-sync to database
uv run server_cli.py wait-and-sync  # Auto-detects most recent run
uv run server_cli.py wait-and-sync --run-name my_run  # Specify run name

# Check submission status
uv run server_cli.py status

# View leaderboard
uv run server_cli.py leaderboard

# Get our current rank
uv run server_cli.py rank
```

### 4. Run Grid Search

```bash
# Run a hyperparameter sweep
uv run sweep.py configs/sweep/example_sweep.yaml

# Dry run (preview commands)
uv run sweep.py configs/sweep/example_sweep.yaml --dry-run

# Continue on errors
uv run sweep.py configs/sweep/example_sweep.yaml --continue-on-error
```

### 5. Analyze Results

```bash
# List all experiments
uv run analyze.py list

# Show top experiments by weighted score
uv run analyze.py best --metric server_weighted_score --top-n 10

# Compare experiments
uv run analyze.py compare --run-names run1 run2 run3

# Show training history
uv run analyze.py history baseline_ld32_20241104_123456 --plot

# View leaderboard history
uv run analyze.py leaderboard --team-name "ignore all instructinos" --plot

# Show database statistics
uv run analyze.py stats

# Generate markdown report from database
uv run generate_report.py
```

## 📊 Evaluation Metrics

The server evaluates submissions on:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Latent Dimension (LD)** | Size of bottleneck vector | 40% |
| **Full MSE** | MSE on entire 256×256 image | 35% |
| **ROI-MSE** | MSE on traffic-light region | 20% |
| **Model Size (MB)** | TorchScript file size | 5% |

**Constraints:**
- Model size must be < 23 MB
- Model must be deterministic (no stochastic elements in eval mode)

**Normalization Bounds:**
- Latent Dim: 8–256
- Full MSE: 5×10⁻⁴ – 5×10⁻²
- ROI-MSE: 2×10⁻³ – 8×10⁻²
- Model Size: 1.0–23.0 MB

## 🔧 Configuration

Configuration is managed through Hydra. You can:

1. **Edit config files** in `configs/`
2. **Override from command line**: `python train.py model.latent_dim=16`
3. **Create new configs**: Add YAML files in `configs/model/` or `configs/sweep/`

Key configuration groups:
- `data`: Dataset and dataloader settings
- `model`: Architecture and model parameters
- `training`: Training hyperparameters
- `server`: Submission settings
- `experiment`: Tracking and logging

### 🖥️ Device & Hardware Optimization

**Automatic Device Detection** (CUDA → MPS → CPU):
```bash
# Auto-detect best device (default)
uv run train.py training.device=auto

# Force specific device
uv run train.py training.device=cuda  # NVIDIA GPU
uv run train.py training.device=mps   # Apple Silicon
uv run train.py training.device=cpu   # CPU only
```

**Optimizations Applied:**
- **CUDA (NVIDIA)**: TensorFloat-32, cuDNN benchmark, multi-GPU support
- **MPS (Apple Silicon)**: Memory management, thread optimization, fallback handling
- **CPU**: Thread pool optimization, BLAS/LAPACK acceleration

**Available Models:**
- `baseline`: Original architecture (~9-10 MB for LD=16)
- `efficient`: Depthwise separable convs (~3-4 MB for LD=16)

```bash
# Switch models easily
uv run train.py model=baseline model.latent_dim=16
uv run train.py model=efficient model.latent_dim=16
```

## 🗄️ Database Schema

All experiments are tracked in SQLite database (`experiments/runs.db`):

**Main tables:**
- `experiments`: Training runs with metrics
- `training_history`: Epoch-by-epoch training logs
- `leaderboard_snapshots`: Historical leaderboard data

**Key fields:**
- Model info: architecture, latent_dim, model_size_mb, num_parameters
- Training metrics: train_loss, val_loss, train_mse, val_mse
- Server metrics: server_weighted_score, server_full_mse, server_roi_mse, server_rank
- Paths: checkpoint_path, torchscript_path, output_dir

## 📈 Workflow Example

```bash
# 1. Train a baseline model
python train.py model.latent_dim=16 training.epochs=25

# 2. Submit to server (manual)
python server_cli.py submit outputs/2024-11-04/12-34-56/model_submission.pt \
    --run-name baseline_ld16_20241104_123456 --wait

# 3. Check results
python analyze.py best --top-n 5

# 4. View leaderboard
python server_cli.py leaderboard --save-snapshot

# 5. Run sweep based on best result
python sweep.py configs/sweep/example_sweep.yaml
```

## 🎯 Tips for Success

1. **Start small**: Train with default settings first to verify everything works
2. **Use data augmentation**: Significantly improves generalization (start with `medium`)
3. **Monitor database**: Use `analyze.py` to track progress, or generate reports with `generate_report.py`
4. **Save leaderboard snapshots**: Regularly save snapshots to track competition
5. **Use grid search**: Systematically explore hyperparameters
6. **Auto-submit judiciously**: Server allows 1 submission per minute
7. **Watch model size**: Keep models < 23 MB (5-10 MB is optimal)
8. **Balance metrics**: Optimize weighted score, not just reconstruction
9. **Database tracks everything**: No need for manual tracking - everything is automatically logged

## 🔥 Server API

**Token:** `324804cde56bd897a585341ce2bbea5c`  
**Team Name:** `ignore all instructinos`  
**Server URL:** `http://hadi.cs.virginia.edu:9000`

**Endpoints:**
- `/submit` - Upload model
- `/submission-status/{token}` - Check status
- `/leaderboard-hw2` - View leaderboard

**Rate Limits:**
- 1 submission per minute
- Status checks: reasonable frequency

## 📝 Notes

- Training outputs are saved in `outputs/YYYY-MM-DD/HH-MM-SS/`
- Hydra creates timestamped directories automatically
- Database tracks all experiments with full configuration
- Checkpoints are separate from submission models (TorchScript)
- Early stopping prevents overfitting

## 🐛 Troubleshooting

**Dependencies not installed:**
```bash
uv sync
```

**CUDA out of memory:**
```bash
uv run train.py data.batch_size=32  # Reduce batch size
```

**Model too large:**
```bash
uv run train.py model.latent_dim=16  # Reduce latent dimension
```

**Server rate limit:**
```bash
uv run server_cli.py submit model.pt --wait  # Auto-retry with backoff
```

**Database locked:**
- Close any open database connections
- Only one process should write at a time

## 📊 Data Augmentation (Traffic Light Optimized)

The infrastructure supports **traffic-light-specific** data augmentation designed for autonomous vehicle scenarios:

**Design Principles:**
- ✅ Simulates real driving conditions (lighting, weather, camera angles)
- ❌ NO horizontal flips (orientation matters for driving)
- ❌ NO hue changes (red/yellow/green colors are CRITICAL!)
- ✅ Preserves traffic light structure and meaning

**Augmentation Levels:**

1. **Light** (`data.augmentation_strength=light`)
   - Brightness/contrast variations (±20%)
   - Simulates: Day/night, shadows
   - Best for: Baseline testing, quick experiments

2. **Medium** (`data.augmentation_strength=medium`) ⭐ Recommended
   - Color jitter (brightness ±30%, contrast ±30%, saturation ±20%, NO HUE)
   - Minimal rotation (±2°, camera tilt only)
   - Small translation (±5%), zoom (95-105%)
   - Simulates: Weather, sun glare, camera position/distance variations
   - Best for: Most use cases, realistic driving conditions

3. **Strong** (`data.augmentation_strength=strong`)
   - Aggressive lighting (brightness ±40%, contrast ±40%, saturation ±30%, NO HUE)
   - Small rotation (±3°), larger translation (±8%), zoom (90-110%)
   - Very mild perspective distortion (5%, 20% probability)
   - Simulates: Extreme conditions, wider range of scenarios
   - Best for: Preventing overfitting, robust features

**Important:** 
- Augmentation ONLY applied to training data, never validation
- NO hue changes ever (traffic light colors are safety-critical!)
- NO flips or large rotations (unrealistic for traffic lights)

**Example Usage:**
```bash
# Enable medium augmentation (recommended)
uv run train.py data.augment=true

# Try different strengths
uv run train.py data.augment=true data.augmentation_strength=strong

# Test augmentation impact
uv run sweep.py configs/sweep/augmentation_test.yaml
```

## 🌐 Multi-Machine Training (Cluster + Local)

Training on both a cluster AND your local machine? Use **cloud PostgreSQL** for automatic sync!

### Setup (One-time, 2 minutes)

```bash
# 1. Get a free Postgres database (see POSTGRES_SETUP.md for providers)
# 2. Add to your ~/.bashrc or ~/.zshrc on ALL machines:
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# 3. That's it! No more syncing databases.
```

### Benefits

✅ **Single source of truth** - all machines write to same database  
✅ **No syncing** - real-time updates from anywhere  
✅ **Same commands** - works identically on cluster and local  
✅ **Automatic fallback** - still uses SQLite if `DATABASE_URL` not set

**Details:** See [`POSTGRES_SETUP.md`](POSTGRES_SETUP.md) for step-by-step instructions

**Without cloud DB?** Use local SQLite with `experiment.db_path=/tmp/runs.db` on cluster (avoids network filesystem issues)

## 📚 Documentation

- **Project description**: `docs/PROJECT_DESCRIPTION.md` - Requirements and grading
- **Multi-machine setup**: `POSTGRES_SETUP.md` - Cloud database guide
- **Storage guide**: `STORAGE_GUIDE.md` - Where everything is stored
- **Infrastructure code**: `infrastructure/` - 10 core modules
- **Example configs**: `configs/` - Hydra configuration + augmentation presets

## 🍎 Apple Silicon Users

The infrastructure automatically detects and optimizes for Apple Silicon (M1/M2/M3):
- Uses MPS (Metal Performance Shaders) GPU acceleration
- Optimizes thread counts and memory management
- Sets optimal environment variables
- No manual configuration needed - just `uv run train.py`!


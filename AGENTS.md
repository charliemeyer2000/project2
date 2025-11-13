# Agent Learnings - Project 2

## Agent Guidelines

**DO NOT:**
- Write temporary test scripts or one-off scripts
- Create summary markdown documents
- Generate unnecessary documentation files

**INSTEAD:**
- Use inline Python code that executes in terminal: `python -c "code here"` or `python << 'EOF' ... EOF`
- Make permanent, reusable changes to existing codebase files
- Update this AGENTS.md if you learn something critical

## Project Overview

**Task:** Build a convolutional autoencoder to compress and reconstruct traffic light images from autonomous vehicle cameras.

**Dataset:** 14,034 RGB images (256×256) from roof-mounted stereo camera, captured under various lighting/weather conditions.

**Constraints:**
- Input/Output: RGB images [B, 3, 256, 256], values in [0,1]
- Model size: < 23 MB
- Submission: TorchScript (.pt) via `torch.jit.script(model)`
- Rate limit: 1 submission per 45 minutes

## Evaluation Metrics

**Raw Metrics (lower is better):**
- **Latent Dim (LD):** Size of bottleneck vector [8-256]
- **Full MSE:** Reconstruction MSE over entire 256×256 frame [5e-4 to 5e-2]
- **ROI MSE:** MSE over traffic light region only [2e-3 to 8e-2]
- **Model Size:** TorchScript file size in MB [1.0 to 23.0]

**Weighted Score (higher is better):**
```
Score = 0.40·LD′ + 0.35·FullMSE′ + 0.20·ROI′ + 0.05·Size′
```
Each metric normalized to [0,1] using: `score′ = 1 − (x − min) / (max − min)`

**Tie-breakers (in order):**
1. Lower Latent Dim
2. Lower Full MSE
3. Lower ROI MSE
4. Lower Model Size
5. Earlier submission time

**Leaderboards:**
- **Public:** Evaluated on full training dataset (visible after submission)
- **Private:** Hidden evaluation dataset (final ranking, revealed at end)

## Architecture Requirements

**Must have:**
- Encoder: Conv blocks → flatten → Linear → latent code `[B, latent_dim]`
- Decoder: Linear → reshape → ConvTranspose blocks → output `[B, 3, 256, 256]`
- Final activation: Sigmoid (outputs in [0,1])
- Forward signature: `model.forward(x)` returns reconstructed image
- Deterministic eval: `model.eval()` must work

**Latent dim discovery (server checks in order):**
1. `model.latent_dim`
2. `model.enc.latent_dim`
3. `model.enc.fc.out_features`
4. **NEW (as of Nov 12):** `model.enc(dummy_input).shape[1]`

## Critical: Server Latent Dim Inference

**THE BUG:** Server infers `latent_dim` by running `encoder(dummy_input).shape[1]`. Our encoder returned a **tuple** `(z, skips)` which broke this check.

**THE FIX:**
```python
# Encoder must return ONLY tensor, not tuple
def forward(self, x):
    # ... compute z ...
    return z  # NOT return z, skips
```

**Server checks in order:**
1. `model.latent_dim`
2. `model.enc.latent_dim` 
3. `model.enc.fc.out_features`
4. **NEW:** `model.enc(dummy).shape[1]` (runs encoder!)

## Core Commands

### Local Development
```bash
cd /Users/charlie/all/UVA/4/F25/ml/projects/project2

# Test model
uv run python -c "from models import get_model; ..."

# Export model
uv run python export_model.py path/to/checkpoint.pth -o output.pt

# Check status
uv run python server_cli.py status
uv run python server_cli.py rank
```

### Cluster Workflow
```bash
# 1. Push code
git add -A && git commit -m "fix" && git push

# 2. SSH to cluster
ssh abs6bd@hadi.cs.virginia.edu
cd all/project2

# 3. Pull updates
git pull

# 4. CRITICAL: Clear Python cache
rm -rf models/__pycache__ infrastructure/__pycache__

# 5. Train
bash a100_run.sh

# 6. Export & Submit
uv run python export_model.py outputs/.../best_model.pth -o model.pt
uv run python server_cli.py submit model.pt
```

## TorchScript Issues

**Problem:** Can't set attributes in `forward()` after `__init__()`

**Wrong:**
```python
def forward(self, x):
    self._last_skips = [h1, h2, h3]  # ❌ TorchScript error
```

**Solution:** Initialize in `__init__()` or avoid storing state entirely
```python
def __init__(self):
    self._last_skips = []  # ✅ OK
```

**Workaround:** Disable skip connections for TorchScript export:
```python
model = get_model('attention', use_skip_connections=False, ...)
```

## Database Syncing

Automatic when using `server_cli.py wait-and-sync`:
```bash
uv run python server_cli.py wait-and-sync --run-name YOUR_RUN_NAME
```

Manual:
```bash
uv run python server_cli.py status  # Just check, no DB update
```

## Improvements Needed

1. **Fix skip connections for TorchScript** - Redesign to avoid storing state in forward()
2. **Better width_mult inference** - Currently hardcoded inference, should store in checkpoint config properly
3. **Simplify export_model.py** - Too much inference logic, should trust checkpoint config more
4. **Add pre-submission validation** - Test that `encoder(dummy).shape[1]` works before submitting
5. **Update training to save correct config** - width_mult, use_skip_connections not saved properly

## Quick Reference

**Model size formula:** `params * 4 / (1024**2)` MB (fp32)

**Target specs:**
- Latent dim: 16
- Width mult: 1.6
- Model size: ~15 MB
- Expected score: 0.92-0.94

**Competition weights:**
- 40% Latent Dim
- 35% Full MSE
- 20% ROI MSE
- 5% Model Size

## Key Insights & Hints

**Architecture choices:**
- Skip connections (U-Net style) improve ROI MSE significantly
- Attention mechanisms help focus on traffic light regions
- GroupNorm > BatchNorm for small batch sizes
- Tanh activation + rescale to [0,1] can work better than Sigmoid

**Training tips:**
- Use combined loss: ROI-weighted MSE + perceptual loss
- High `lambda_roi` (10-15) emphasizes traffic light quality
- Smaller ROI size (0.2) focuses on tighter regions
- LR warmup prevents early instability
- Gradient clipping (max_norm=1.0) stabilizes training

**Model sizing:**
- `width_mult` controls channel scaling (1.6 → ~15 MB)
- Latent dim has biggest impact on score (16 is sweet spot)
- Target: 14-16 MB models with LD=16

**Common pitfalls:**
- Encoder returning tuple instead of tensor (breaks server check)
- TorchScript can't set new attributes in forward()
- Python cache (`__pycache__`) can cause old code to run
- Config in checkpoint may be wrong (use state_dict inference)


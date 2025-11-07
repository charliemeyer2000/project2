#!/usr/bin/env python3
"""Re-export attention model from checkpoint with fixed latent_dim visibility."""

import torch
from models import get_model
from infrastructure.checkpoint import save_torchscript

print("=" * 60)
print("Re-exporting Attention Model with Fixed latent_dim")
print("=" * 60)

# Load best checkpoint
checkpoint_path = 'outputs/2025-11-06/20-26-07/checkpoints/best_checkpoint.pth'
print(f"\n📂 Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Create model with CORRECT architecture
print("\n🔨 Creating attention model with latent_dim=16...")
model = get_model('attention', latent_dim=16)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Verify latent_dim is accessible (all 3 patterns the server checks)
print("\n✅ Verifying latent_dim is discoverable:")
print(f"   Pattern 1: model.latent_dim = {model.latent_dim}")
print(f"   Pattern 2: model.enc.latent_dim = {model.enc.latent_dim}")
print(f"   Pattern 3: model.enc.fc.out_features = {model.enc.fc.out_features}")

# Export to TorchScript
output_path = 'outputs/2025-11-06/20-26-07/model_FINAL_FIXED.pt'
print(f"\n📦 Exporting to TorchScript: {output_path}")
save_torchscript(model, output_path, verify=True)

print("\n" + "=" * 60)
print("✅ SUCCESS! Model exported with all latent_dim patterns!")
print("=" * 60)
print(f"\n📤 Next steps:")
print(f"   1. Submit: uv run server_cli.py submit {output_path}")
print(f"   2. Sync:   uv run server_cli.py wait-and-sync")
print()


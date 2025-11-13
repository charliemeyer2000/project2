#!/bin/bash
# Update cluster with the FINAL fix for encoder tuple return issue
#
# This fixes the critical bug where AttentionEncoder.forward() returned
# a tuple (z, skips) but the server now infers latent_dim by running
# encoder(input).shape[1], which fails on tuples.
#
# Usage: ./update_cluster_final.sh

set -e

CLUSTER_USER="abs6bd"
CLUSTER_HOST="hadi.cs.virginia.edu"
PROJECT_DIR="all/project2"

echo "============================================================"
echo "🔧 FINAL FIX: Update encoder to return only tensor"
echo "============================================================"
echo ""
echo "Issue: Server changed to infer latent_dim from encoder output"
echo "  Old: Checked attributes (latent_dim, enc.latent_dim, enc.fc.out_features)"
echo "  New: Runs encoder(input).shape[1] to infer latent_dim"
echo ""
echo "Problem: AttentionEncoder returned tuple (z, skips)"
echo "Solution: Return only z, store skips in self._last_skips"
echo ""

# Copy fixed attention_autoencoder.py
echo "1️⃣ Copying fixed models/attention_autoencoder.py..."
scp models/attention_autoencoder.py ${CLUSTER_USER}@${CLUSTER_HOST}:~/${PROJECT_DIR}/models/

echo "   ✅ Fixed encoder uploaded"
echo ""
echo "============================================================"
echo "✅ Files updated on cluster!"
echo "============================================================"
echo ""
echo "Now SSH to cluster and run:"
echo ""
echo "  ssh ${CLUSTER_USER}@${CLUSTER_HOST}"
echo "  cd ${PROJECT_DIR}"
echo ""
echo "  # Clear Python cache"
echo "  rm -rf models/__pycache__"
echo ""
echo "  # Re-export with fixed encoder"
echo "  uv run python export_model.py outputs/2025-11-12/00-22-02/checkpoints/best_model.pth -o outputs/2025-11-12/00-22-02/model_FINAL_FIX.pt"
echo ""
echo "  # Submit"
echo "  uv run python server_cli.py submit outputs/2025-11-12/00-22-02/model_FINAL_FIX.pt"
echo ""


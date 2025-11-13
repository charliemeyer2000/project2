#!/bin/bash
# Script to update cluster with fixed export code
#
# This script copies the fixed export_model.py to the cluster and re-exports
# the model with proper width_mult inference and verification.
#
# Usage (from local machine):
#   ./update_cluster.sh

set -e  # Exit on error

CLUSTER_USER="abs6bd"
CLUSTER_HOST="hadi.cs.virginia.edu"
PROJECT_DIR="all/project2"

echo "============================================================"
echo "Updating cluster with fixed export code"
echo "============================================================"

# 1. Copy fixed export_model.py
echo ""
echo "1️⃣ Copying fixed export_model.py..."
scp export_model.py ${CLUSTER_USER}@${CLUSTER_HOST}:~/${PROJECT_DIR}/

echo "   ✅ export_model.py updated"

# 2. Copy fixed attention_autoencoder.py (has latent_dim fix)
echo ""
echo "2️⃣ Copying fixed attention_autoencoder.py..."
scp models/attention_autoencoder.py ${CLUSTER_USER}@${CLUSTER_HOST}:~/${PROJECT_DIR}/models/

echo "   ✅ attention_autoencoder.py updated"

# 3. Instructions for cluster
echo ""
echo "============================================================"
echo "✅ Files updated on cluster!"
echo "============================================================"
echo ""
echo "Now SSH to the cluster and run:"
echo ""
echo "  ssh ${CLUSTER_USER}@${CLUSTER_HOST}"
echo "  cd ${PROJECT_DIR}"
echo "  uv run python export_model.py outputs/2025-11-12/00-22-02/checkpoints/best_model.pth"
echo ""
echo "This will:"
echo "  - Automatically infer width_mult=1.6 from weights"
echo "  - Verify latent_dim is accessible"
echo "  - Test inference"
echo "  - Create model_submission.pt"
echo ""
echo "Then submit:"
echo "  uv run python server_cli.py submit outputs/2025-11-12/00-22-02/checkpoints/model_submission.pt"
echo ""


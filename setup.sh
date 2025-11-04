#!/bin/bash
set -e

echo "=========================================="
echo "Project2 Setup Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root (where pyproject.toml is)"
    exit 1
fi

# 1. Install dependencies
echo ""
echo "📦 Installing dependencies with uv..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv sync

# 2. Setup dataset
echo ""
echo "📂 Setting up dataset..."

if [ -d "data/training_dataset" ] && [ "$(ls -A data/training_dataset/*.jpg 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "✅ Dataset already exists ($(ls data/training_dataset/*.jpg 2>/dev/null | wc -l) images)"
else
    mkdir -p data
    
    # Check if zip file exists in current directory
    if [ -f "training_dataset.zip" ]; then
        echo "📦 Found training_dataset.zip, extracting..."
        unzip -q training_dataset.zip -d data/
        echo "✅ Dataset extracted"
    # Check if zip exists in parent directory
    elif [ -f "../training_dataset.zip" ]; then
        echo "📦 Found training_dataset.zip in parent directory, extracting..."
        unzip -q ../training_dataset.zip -d data/
        echo "✅ Dataset extracted"
    else
        echo "⚠️  Dataset zip not found!"
        echo ""
        echo "Please do ONE of the following:"
        echo "  1. Copy training_dataset.zip to this directory, then run:"
        echo "     unzip training_dataset.zip -d data/"
        echo ""
        echo "  2. If on cluster, copy dataset from your local machine:"
        echo "     scp -r /path/to/local/data/training_dataset username@cluster:/path/to/project2/data/"
        echo ""
        echo "  3. Download dataset (if you have a URL):"
        echo "     wget <DATASET_URL> -O training_dataset.zip"
        echo "     unzip training_dataset.zip -d data/"
        echo ""
        exit 1
    fi
fi

# Verify dataset
NUM_IMAGES=$(find data/training_dataset -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
if [ "$NUM_IMAGES" -lt 1000 ]; then
    echo "❌ Error: Only found $NUM_IMAGES images. Expected ~14000"
    exit 1
fi
echo "✅ Dataset verified: $NUM_IMAGES images"

# 3. Create necessary directories
echo ""
echo "📁 Creating output directories..."
mkdir -p experiments
mkdir -p outputs

# 4. Test device detection
echo ""
echo "🔍 Testing device detection..."
uv run python -c "
from infrastructure.device import get_device, print_device_info
device = get_device('auto')
print_device_info(device)
"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo "  # Test training (1 epoch):"
echo "  uv run train.py training.epochs=1"
echo ""
echo "  # Full training run:"
echo "  uv run train.py model=efficient model.latent_dim=16"
echo ""
echo "  # With auto-submission:"
echo "  uv run train.py model=efficient server.auto_submit=true"
echo ""


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

# Check if dataset already exists
if [ -d "data/training_dataset" ] && [ "$(ls -A data/training_dataset/*.jpg 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "✅ Dataset already exists ($(ls data/training_dataset/*.jpg 2>/dev/null | wc -l) images)"
else
    mkdir -p data
    
    # Check if zip file exists in current directory
    if [ -f "training_dataset.zip" ]; then
        echo "📦 Found training_dataset.zip locally, extracting..."
        unzip -q training_dataset.zip -d data/
        rm training_dataset.zip  # Clean up after extraction
        echo "✅ Dataset extracted"
    # Check if zip exists in parent directory
    elif [ -f "../training_dataset.zip" ]; then
        echo "📦 Found training_dataset.zip in parent directory, extracting..."
        unzip -q ../training_dataset.zip -d data/
        echo "✅ Dataset extracted"
    else
        # Try to download from server
        echo "📥 Dataset not found locally, downloading from server..."
        
        # Use provided URL or default to course server
        DOWNLOAD_URL="${DATASET_URL:-http://hadi.cs.virginia.edu:9000/download/train-dataset-hw2}"
        
        echo "📥 Downloading from: $DOWNLOAD_URL"
        
        # Try wget first, then curl
        if command -v wget &> /dev/null; then
            wget --progress=bar:force:noscroll "$DOWNLOAD_URL" -O training_dataset.zip
            DOWNLOAD_STATUS=$?
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar -o training_dataset.zip "$DOWNLOAD_URL"
            DOWNLOAD_STATUS=$?
        else
            echo "❌ Error: Neither wget nor curl found. Cannot download."
            echo ""
            echo "Install wget or curl, or manually download:"
            echo "  scp username@your-mac:/path/to/training_dataset.zip ."
            echo "  ./setup.sh"
            exit 1
        fi
        
        # Check if download succeeded
        if [ $DOWNLOAD_STATUS -ne 0 ] || [ ! -f "training_dataset.zip" ]; then
            echo "❌ Download failed!"
            echo ""
            echo "Alternative options:"
            echo "  1. Copy from local machine:"
            echo "     scp username@your-mac:/path/to/training_dataset.zip ."
            echo "     ./setup.sh"
            echo ""
            echo "  2. Set custom URL and retry:"
            echo "     export DATASET_URL='https://your-custom-url/training_dataset.zip'"
            echo "     ./setup.sh"
            exit 1
        fi
        
        echo "📦 Extracting dataset..."
        unzip -q training_dataset.zip -d data/
        
        # Clean up __MACOSX folder if it exists (from macOS zips)
        if [ -d "data/training_dataset/__MACOSX" ]; then
            echo "🧹 Removing __MACOSX folder..."
            rm -rf data/training_dataset/__MACOSX
        fi
        
        rm training_dataset.zip  # Clean up zip file
        echo "✅ Dataset downloaded and extracted"
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


#!/bin/bash

# Set the environment name
ENV_NAME=asl-gpu

# Load conda (adjust if your conda path differs)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "❌ Conda not found. Please install Miniforge or Miniconda for ARM64."
    exit 1
fi

# Check if environment exists, create if not
if ! conda info --envs | grep -q "^$ENV_NAME"; then
    echo "🔧 Creating environment '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=3.9
fi

# Activate the environment
conda activate $ENV_NAME

# Install required packages
echo "📦 Installing dependencies..."
conda install -y numpy=1.24 pandas matplotlib tensorflow=2.12 opencv

# (Optional) If you're using pip for extra packages
# pip install some_package

echo "✅ Environment setup complete. To activate it later, run:"
echo "conda activate $ENV_NAME"

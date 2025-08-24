#!/bin/bash

# Auto setup script using UV package manager
# This script sets up the environment and installs dependencies for TabularML

set -e  # Exit on any error

echo "🚀 Setting up TabularML environment with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ UV version: $(uv --version)"

# Initialize UV project if not already initialized
if [ ! -f "uv.lock" ]; then
    echo "🔧 Initializing UV project..."
    uv sync
fi

echo "📚 Installing dependencies..."
uv sync

echo "🎉 Environment setup complete!"
echo ""
echo "Available commands:"
echo "  uv run streamlit run ui.py          # Run the Streamlit UI"
echo "  uv run python pipeline.py          # Run the ML pipeline directly"
echo "  uv run python -m pytest            # Run tests (if available)"
echo ""
echo "To activate the environment manually:"
echo "  source .venv/bin/activate"
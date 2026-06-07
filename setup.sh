#!/bin/bash
# setup.sh — Run this once to set up the environment
# Usage: bash setup.sh

set -e

echo "================================================"
echo "DQN Feature Selection — Project Setup"
echo "================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.9+ first."
    exit 1
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYVER"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Setup complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo ""
echo "NEXT STEPS:"
echo "  1. Download UNSW-NB15 CSVs from:"
echo "     https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo "  2. Place all 4 CSV files in the data/ folder"
echo "  3. Run in order:"
echo "     python src/01_preprocess.py"
echo "     python src/02_baselines.py"
echo "     python src/03_train_dqn.py"
echo "     python src/04_evaluate.py"
echo "     python src/05_plots.py"
echo "================================================"

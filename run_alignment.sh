#!/bin/bash

# Set project root (assumes script is in the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure PYTHONPATH is set correctly
export PYTHONPATH="$PROJECT_ROOT"

# Conda environment name
CONDA_ENV="atomicity"

# Properly initialize Conda for non-interactive shells
eval "$(conda shell.bash hook)" || { echo "Conda initialization failed. Ensure Conda is installed."; exit 1; }

# Activate the Conda environment
conda activate "$CONDA_ENV" || { echo "Failed to activate Conda environment: $CONDA_ENV"; exit 1; }

# Set Python executable (adjust if needed)
PYTHON_EXEC="python"  # Change this if you're using a virtual env

# Define script path
SCRIPT_PATH="$PROJECT_ROOT/src/metrics/main_alignment.py"

# Define log file (overwrites on each run)
LOG_FILE="$PROJECT_ROOT/alignment_log.txt"

# Run script and save all output (stdout + stderr) to the log file
"$PYTHON_EXEC" "$SCRIPT_PATH" 2>&1 | tee "$LOG_FILE"

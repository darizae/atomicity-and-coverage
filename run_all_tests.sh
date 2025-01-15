#!/bin/bash
set -e  # Exit immediately on error

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

# Function to remove generated files
cleanup() {
  echo "Cleaning up generated files..."
  rm -f rose_datasets.json.gz
  rm -f rose_datasets.json
  rm -f rose_datasets_small.json.gz
  rm -f rose_datasets_small.json
  rm -f local_rose_datasets.json.gz
  echo "Cleanup complete."
}

# Trap to ensure cleanup runs even if the script exits due to an error
trap cleanup EXIT

# Run pytest for tests
echo "Running unit tests with pytest..."
pytest

# Run generate_claims.py with --small_test flag
echo "Running generate_claims.py with --small_test..."
python src/claims/generate_claims.py --small_test

# Run main.py to check for exceptions
echo "Running main.py..."
python src/main.py

echo "All tests completed successfully!"

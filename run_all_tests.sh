#!/bin/bash
set -e  # Exit immediately on error

# Set PYTHONPATH to include the project root
export PYTHONPATH=$(pwd)

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

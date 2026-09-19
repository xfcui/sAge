#!/usr/bin/env bash
set -euo pipefail

# Portable wrapper for the Python cross-validation runner.
# Example: bash test1-5x-tissue.sh --data-dir ./data --output-dir ./outputs
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.20}"
python3 "$(dirname "$0")/../run_cross_validation.py" "$@"

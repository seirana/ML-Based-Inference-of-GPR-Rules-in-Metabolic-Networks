#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper around the portable Python orchestrator.
python -m run_pipeline "$@"

#!/usr/bin/env bash
set -euo pipefail

if python -m pip show torch >/dev/null 2>&1; then
  for test_file in test_*.py; do
    echo "==> ${test_file}"
    python "${test_file}"
  done

  echo "==> validate_api.py"
  python validate_api.py
else
  echo "PyTorch is not installed; running dependency-light checks only."
  python test_hardening.py
fi

#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" scripts/prepare_brep_public_data.py \
  --datasets prm800k ultrafeedback gsm8k hellaswag svamp mathqa math500 amc23 truthfulqa \
  "$@"

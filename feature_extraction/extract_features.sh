#!/usr/bin/env bash
set -euo pipefail

TRIDENT_DIR=$1
WSI_DIR=$2
JOB_DIR=$3
ENCODER=${4:-uni_v1}

python "$TRIDENT_DIR/run_batch_of_slides.py" --task all \
  --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
  --patch_encoder "$ENCODER" --mag 10 --patch_size 256 --overlap 0

#!/usr/bin/env bash
# AlignVid on FramePack (v1) — Image-to-Video.
# Usage: edit the paths below, then run:  bash evaluation/generation/run_framepack_v1.sh
#
# NOTE: FramePack (v1) uses a different foreground-block preset than F1.
# In models/FramePack/diffusers_helper/models/hunyuan_video_packed.py set:
#     ALIGNVID_VARIANT = 'framepack'
set -euo pipefail

# --- paths (edit these) ---
DATA_DIR=/path/to/OmitI2V                  # OmitI2V dataset root (image-path is relative to this)
PROMPT_FILE="$DATA_DIR/meta.json"          # OmitI2V metadata
OUTPUT_DIR=./results/framepack_v1          # videos are saved as {id}.mp4

# AlignVid: --control_scale is gamma (1.35); use 1.0 for the baseline.
cd "$(dirname "$0")/../../models/FramePack"
python inference_v1_json.py \
    --data_dir "$DATA_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --control_scale 1.35

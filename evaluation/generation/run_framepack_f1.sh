#!/usr/bin/env bash
# AlignVid on FramePack-F1 — Image-to-Video.
# Usage: edit the paths below, then run:  bash evaluation/generation/run_framepack_f1.sh
set -euo pipefail

# --- paths (edit these) ---
DATA_DIR=/path/to/OmitI2V                  # OmitI2V dataset root (image-path is relative to this)
PROMPT_FILE="$DATA_DIR/meta.json"          # OmitI2V metadata
OUTPUT_DIR=./results/framepack_f1          # videos are saved as {id}.mp4

# FramePack-F1 weights are loaded by the script's HF paths; see models/FramePack/README.md.
# AlignVid: --control_scale is gamma (1.35); use 1.0 for the baseline, add --blur_img for the blur baseline.
cd "$(dirname "$0")/../../models/FramePack"

# single GPU
python inference_f1_json.py \
    --data_dir "$DATA_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --control_scale 1.35

# multi-GPU (e.g. 4 GPUs), uncomment to use:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --nnodes=1 --master_port=39502 \
#     inference_f1_json.py \
#     --data_dir "$DATA_DIR" --prompt_file "$PROMPT_FILE" --output_dir "$OUTPUT_DIR" \
#     --control_scale 1.35

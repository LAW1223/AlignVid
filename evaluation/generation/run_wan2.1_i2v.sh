#!/usr/bin/env bash
# AlignVid on Wan2.1 — Image-to-Video.
# Usage: edit the paths below, then run:  bash evaluation/generation/run_wan2.1_i2v.sh
set -euo pipefail

# --- paths (edit these) ---
CKPT_DIR=/path/to/model_weights/Wan2.1-I2V-14B-480P   # Wan2.1-I2V-14B weights
DATA_DIR=/path/to/OmitI2V                             # OmitI2V dataset root (image-path is relative to this)
PROMPT_FILE="$DATA_DIR/meta.json"                     # OmitI2V metadata
OUTPUT_DIR=./results/wan2.1                           # videos are saved as {id}.mp4

# --- AlignVid setting (gamma = 1.35; set ALL scales to 1.0 to get the baseline) ---
cd "$(dirname "$0")/../../models/Wan2.1"
python generate_i2v_json.py \
    --task i2v-14B \
    --size 480*832 \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir "$DATA_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --self_attn_scale 1.0 \
    --cross_attn_q_image_scale 1.35 \
    --cross_attn_k_image_scale 1.0 \
    --cross_attn_q_text_scale 1.0 \
    --cross_attn_k_text_scale 1.35 \
    --sample_guide_scale 1.0

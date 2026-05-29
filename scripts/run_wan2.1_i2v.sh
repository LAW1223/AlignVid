#!/usr/bin/env bash
# AlignVid quick demo — Wan2.1 (Image-to-Video).
# Edit the variables below, then run:  bash scripts/run_wan2.1_i2v.sh
set -e

IMAGE=/path/to/your_image.png                    # input image
PROMPT="the cat puts on a pair of sunglasses"    # the edit you want
OUTPUT_DIR=./results                             # output folder (video: sample.mp4)
CKPT_DIR=/path/to/Wan2.1-I2V-14B-480P            # Wan2.1-I2V-14B weights
GAMMA=1.35                                        # AlignVid strength (1.0 = baseline)

IMAGE=$(realpath "$IMAGE"); mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
PROMPT_FILE=$(mktemp --suffix .json)
echo "[{\"id\":\"sample\",\"prompt\":\"$PROMPT\",\"image-path\":\"$(basename "$IMAGE")\",\"resolution\":[832,480]}]" > "$PROMPT_FILE"

cd "$(dirname "$0")/../models/Wan2.1"
python generate_i2v_json.py \
    --task i2v-14B --size 480*832 \
    --ckpt_dir "$CKPT_DIR" \
    --data_dir "$(dirname "$IMAGE")" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --sample_guide_scale 1.0 \
    --cross_attn_q_image_scale "$GAMMA" \
    --cross_attn_k_text_scale "$GAMMA"

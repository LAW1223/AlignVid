#!/usr/bin/env bash
# AlignVid quick demo — FramePack v1 (Image-to-Video).
# Edit the variables below, then run:  bash scripts/run_framepack_v1.sh
# NOTE: for v1, set ALIGNVID_VARIANT = 'framepack' in
#   models/FramePack/diffusers_helper/models/hunyuan_video_packed.py
set -e

IMAGE=/path/to/your_image.png                    # input image
PROMPT="the cat puts on a pair of sunglasses"    # the edit you want
OUTPUT_DIR=./results                             # output folder (video: sample.mp4)
GAMMA=1.35                                        # AlignVid strength (1.0 = baseline)

IMAGE=$(realpath "$IMAGE"); mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
PROMPT_FILE=$(mktemp --suffix .json)
echo "[{\"id\":\"sample\",\"prompt\":\"$PROMPT\",\"image-path\":\"$(basename "$IMAGE")\"}]" > "$PROMPT_FILE"

cd "$(dirname "$0")/../models/FramePack"
python inference_v1_json.py \
    --data_dir "$(dirname "$IMAGE")" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --control_scale "$GAMMA"

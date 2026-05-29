#!/usr/bin/env bash
# VQA-based semantic-fidelity evaluation (Qwen2.5-VL) — the primary metric.
# Run in the environment that has transformers + qwen_vl_utils installed.
#
# Usage: bash evaluation/eval_vqa.sh <video_dir> [name]
#   <video_dir>  folder of generated videos named {id}.mp4 (e.g. sample_0.mp4)
#   [name]       label for the output file (default: results)
set -euo pipefail

# ----- edit these -----
META=/path/to/OmitI2V/meta.json                 # OmitI2V metadata (with questions)
QWEN_MODEL=/path/to/Qwen2.5-VL-32B-Instruct     # VQA model (LOCAL folder; download it first)
OUT_DIR=./eval_results                          # where results are written
# ----------------------

VIDEO_DIR=${1:?"usage: bash eval_vqa.sh <video_dir> [name]"}
NAME=${2:-results}

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "===== VQA-based (semantic fidelity): $NAME ====="
python "$EVAL_DIR/vqa_based/qwen.py" \
    --json_file "$META" \
    --video_dir "$VIDEO_DIR" \
    --model_path "$QWEN_MODEL" \
    --output_file "$OUT_DIR/${NAME}_vqa.json"
echo "Done -> $OUT_DIR/${NAME}_vqa.json"

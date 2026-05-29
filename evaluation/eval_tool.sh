#!/usr/bin/env bash
# Tool-based visual-quality evaluation (VBench).
# Default dimensions: Dynamic Degree + Aesthetic Quality (the two reported in the paper).
# Requires VBench installed under tool_based/ (see README) and its conda env.
#
# Usage: bash evaluation/eval_tool.sh <video_dir> [name]
#   <video_dir>  folder of generated videos named {id}.mp4
#   [name]       label for the output file (default: results)
set -euo pipefail

# ----- edit this -----
OUT_DIR=./eval_results          # where results are written
# ---------------------

VIDEO_DIR=${1:?"usage: bash eval_tool.sh <video_dir> [name]"}
NAME=${2:-results}

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "===== Tool-based (VBench: Dynamic Degree + Aesthetic Quality): $NAME ====="
# Run from tool_based/ so the vbench2_beta_i2v package is importable.
# To evaluate more dimensions, append e.g.:
#   --dimension dynamic_degree aesthetic_quality subject_consistency background_consistency
( cd "$EVAL_DIR/tool_based" && python evaluate_i2v.py \
    --videos_path "$VIDEO_DIR" \
    --output_path "$OUT_DIR" \
    --name "$NAME" )
echo "Done -> $OUT_DIR/${NAME}_eval_results.json"

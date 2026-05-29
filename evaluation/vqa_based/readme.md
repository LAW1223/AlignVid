# VQA-based semantic-fidelity evaluation

Uses Qwen2.5-VL to answer each sample's yes/no questions about the generated
video and reports per-sample (macro-averaged) accuracy. The model only sees the
video — it is never given the prompt.

See the top-level [`evaluation/README.md`](../README.md) for the full workflow
and the convenience script `evaluation/eval_vqa.sh`.

## Environment

```bash
conda create -n omit python=3.10
conda activate omit
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r ../requirements.txt
```

## Run

```bash
python qwen.py \
    --json_file /path/to/OmitI2V/meta.json \
    --video_dir /path/to/generated_videos \
    --model_path /path/to/Qwen2.5-VL-32B-Instruct \
    --output_file evaluation_results.json

# quick test on the first 3 samples:
python qwen.py ... --max_items 3
```

| Argument | Description |
|---|---|
| `--json_file` | OmitI2V `meta.json` (each sample carries its `questions`) |
| `--video_dir` | folder of generated videos, named `{id}.mp4` |
| `--model_path` | path to Qwen2.5-VL |
| `--output_file` | results JSON |
| `--fps` | video sampling frame rate (default 8.0) |
| `--max_items` | limit number of samples (for testing) |

## Output

The results JSON contains overall accuracy, per-category (modification /
addition / deletion), per-domain, and per-dimension breakdowns, plus the
per-video details.

> The benchmark already ships the `questions` in `meta.json`, so no
> question-generation step is needed.
> The model analyses the video content only and is never given any prompt hint.

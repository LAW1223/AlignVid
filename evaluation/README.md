# Evaluation

OmitI2V is evaluated with two complementary tools:

| Track | Folder | Measures |
|---|---|---|
| **VQA-based** (primary semantic metric) | [`vqa_based/`](./vqa_based) | Whether the requested edit is realized — Qwen2.5-VL answers the per-sample yes/no questions; we report accuracy. |
| **Tool-based** (visual quality) | [`tool_based/`](./tool_based) | VBench-style metrics: dynamic degree, aesthetic quality, subject/background consistency, motion smoothness. |

The benchmark, including the questions used by the VQA track, is on Hugging Face:
**https://huggingface.co/datasets/AIPeanutman/OmitI2V**

All scripts expect generated videos named `{id}.mp4` (the `id` field of each sample, e.g. `sample_0.mp4`). Replace every `/path/to/...` with your own paths.

> **Generate the videos first.** The batch scripts in
> [`generation/`](./generation) run AlignVid over the whole OmitI2V benchmark and
> produce these `{id}.mp4` files, ready to be scored by the tools below.

## Quick start

Two convenience scripts. Run each **in its own conda env** — VQA needs transformers/Qwen,
tool-based needs VBench. Edit the paths at the top of each, then:

```bash
# VQA-based semantic fidelity (primary)
bash evaluation/eval_vqa.sh  /path/to/generated_videos my_run

# Tool-based visual quality (Dynamic Degree + Aesthetic Quality by default)
bash evaluation/eval_tool.sh /path/to/generated_videos my_run
```

Results are written to `./eval_results/`. The individual commands behind the scripts are documented below.

## Setup

```bash
conda create -n omit python=3.10
conda activate omit

# PyTorch (adjust the CUDA version as needed)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

cd evaluation
pip install -r requirements.txt
```

## 1. VQA-based semantic fidelity (primary)

Uses [Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) to answer each sample's questions about the generated video and reports accuracy. The model only sees the video — it is never given the prompt.

The script loads the model with `local_files_only=True`, so **download it to a local folder first** and point `--model_path` at that folder (a bare HF repo id will not work):

```bash
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct --local-dir ./Qwen2.5-VL-32B-Instruct
```

```bash
cd vqa_based
python qwen.py \
    --json_file /path/to/OmitI2V/meta.json \
    --video_dir /path/to/generated_videos \
    --model_path /path/to/Qwen2.5-VL-32B-Instruct \
    --output_file results_vqa.json
# quick test on the first 3 samples:  python qwen.py ... --max_items 3
```

| Argument | Description |
|---|---|
| `--json_file` | OmitI2V `meta.json` (each sample carries its `questions`) |
| `--video_dir` | folder of generated videos, named `{id}.mp4` |
| `--model_path` | path to Qwen2.5-VL |
| `--output_file` | results JSON |
| `--fps` | sampling frame rate (default 8.0) |
| `--max_items` | limit number of samples (for testing) |

Accuracy is **per-sample (macro-averaged)**: each sample is weighted equally regardless of how many questions it has (samples carry 2–6 questions), so question-heavy samples do not dominate the score. The output reports overall accuracy plus breakdowns by main-category (modification / addition / deletion) and domain (both per-sample macro), and by question dimension (e.g. `object_presence`, `action_correctness`; computed per question).

> The benchmark already ships with the `questions` in `meta.json`, so no question-generation step is needed.

## 2. Tool-based visual quality (VBench metrics)

VBench is **not** bundled here — install it from upstream into `tool_based/`:

```bash
cd tool_based
git clone https://github.com/Vchitect/VBench.git
cp -r VBench/vbench2_beta_i2v/ .
rm -rf VBench
```

Then run:

```bash
python evaluate_i2v.py --videos_path /path/to/generated_videos --output_path ./results --name my_run
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--videos_path` | yes | – | folder of videos to evaluate |
| `--output_path` | no | `./evaluation_i2v_results/` | output directory (writes `{name}_eval_results.json`) |
| `--name` | no | `results` | label used in the output filename |
| `--dimension` | no | `dynamic_degree aesthetic_quality` | VBench dimensions to evaluate (space-separated) |
| `--force_ratio` | no | auto | force an aspect ratio (e.g. `16-9`, `1-1`, `4-3`); otherwise the nearest common ratio is auto-detected |
| `--temp_folder` | no | `./temp_resized_videos/` | temp folder for resized videos |
| `--keep_temp` | no | `False` | keep resized videos after evaluation |
| `--full_json_dir` | no | `tool_based/vbench2_beta_i2v/vbench2_i2v_full_info.json` | VBench prompt/dimension config |

By **default** only the two metrics reported in the paper are computed: **Dynamic Degree** and **Aesthetic Quality**. Add others (e.g. `subject_consistency`, `background_consistency`, `motion_smoothness`) via `--dimension`. On first run VBench downloads the backbone models needed for the selected dimensions.

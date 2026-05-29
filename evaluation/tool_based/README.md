# Tool-based visual-quality evaluation (VBench)

`evaluate_i2v.py` scores generated videos with VBench-style metrics. By
**default** it computes the two dimensions reported in the paper —
**Dynamic Degree** and **Aesthetic Quality** — and reports the raw scores
(no artificial normalization).

See the top-level [`evaluation/README.md`](../README.md) for the full workflow
and the `evaluation/eval_tool.sh` convenience wrapper.

## Install VBench

VBench is **not** vendored here — install it from upstream into this folder:

```bash
git clone https://github.com/Vchitect/VBench.git
cp -r VBench/vbench2_beta_i2v/ .
rm -rf VBench
```

On first run, VBench downloads the backbone models needed for the selected
dimensions (cached locally).

## Run

```bash
python evaluate_i2v.py \
    --videos_path /path/to/generated_videos \
    --output_path ./results \
    --name my_run
```

Results are written to `<output_path>/<name>_eval_results.json`.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--videos_path` | yes | – | folder of videos to evaluate |
| `--output_path` | no | `./evaluation_i2v_results/` | output directory |
| `--name` | no | `results` | label used in the output filename |
| `--dimension` | no | `dynamic_degree aesthetic_quality` | VBench dimensions (space-separated) |
| `--force_ratio` | no | auto | force an aspect ratio (e.g. `16-9`, `1-1`, `4-3`); else nearest is auto-detected |
| `--temp_folder` | no | `./temp_resized_videos/` | temp folder for resized videos |
| `--keep_temp` | no | `False` | keep resized videos after evaluation |
| `--full_json_dir` | no | `vbench2_beta_i2v/vbench2_i2v_full_info.json` | VBench prompt/dimension config |

To evaluate more dimensions, pass them explicitly, e.g.:

```bash
python evaluate_i2v.py --videos_path /path/to/videos \
    --dimension dynamic_degree aesthetic_quality subject_consistency background_consistency motion_smoothness
```

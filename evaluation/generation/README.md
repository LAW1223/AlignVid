# Benchmark video generation

These scripts run AlignVid inference over the **entire OmitI2V benchmark**
(`meta.json`) to produce the `{id}.mp4` videos that the evaluation tools in
[`../`](../) consume. This is **step 1** of reproducing the paper numbers:

```
evaluation/generation/run_*.sh   →   {id}.mp4 for every sample
            │
            ▼
evaluation/eval_vqa.sh / eval_tool.sh   →   metrics
```

> Want to generate a video from **your own** single image + prompt instead of the
> whole benchmark? Use the user CLI in [`../../scripts/`](../../scripts).

Edit the path variables at the top of each script, then run it from the
repository root, e.g.:

```bash
bash evaluation/generation/run_wan2.1_i2v.sh
```

| Script | Backbone | AlignVid knob |
|---|---|---|
| `run_wan2.1_i2v.sh` | Wan2.1 (I2V) | `--cross_attn_q_image_scale 1.35 --cross_attn_k_text_scale 1.35` |
| `run_framepack_f1.sh` | FramePack-F1 | `--control_scale 1.35` |
| `run_framepack_v1.sh` | FramePack (v1) | `--control_scale 1.35` (set `ALIGNVID_VARIANT='framepack'`) |

Defaults reproduce the paper's main setting (γ = 1.35). To get the **baseline**,
set every scale to `1.0` (Wan2.1) or `--control_scale 1.0` (FramePack). Generated
videos are written as `{id}.mp4`, ready for the tools in [`../`](../).

Get the OmitI2V dataset from <https://huggingface.co/datasets/AIPeanutman/OmitI2V>
and point `DATA_DIR` at its root. See the top-level [README](../../README.md) for
full flag documentation and the AlignVid configuration switches.

# Quick inference

Generate a video from your own image + prompt with AlignVid. Open the script for
your backbone, edit the few variables at the top (`IMAGE`, `PROMPT`, `OUTPUT_DIR`,
weights path), then run it — the result is written to `<OUTPUT_DIR>/sample.mp4`.

```bash
bash scripts/run_framepack_f1.sh   # FramePack-F1
bash scripts/run_framepack_v1.sh   # FramePack v1  (also set ALIGNVID_VARIANT='framepack')
bash scripts/run_wan2.1_i2v.sh     # Wan2.1 (needs CKPT_DIR)
```

`GAMMA` is the AlignVid strength (default **1.35**); set it to `1.0` for the
unmodified baseline.

> **Reproducing the OmitI2V benchmark?** Use the batch scripts in
> [`../evaluation/generation/`](../evaluation/generation) instead.

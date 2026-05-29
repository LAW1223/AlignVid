<div align="center">

# 🎬 AlignVid

### A few lines of code to make models follow the prompt better

*Taming Visual Dominance via Training-Free Attention Modulation*

**Official implementation — ICML 2026**

[![ICML 2026](https://img.shields.io/badge/ICML-2026-7c3aed.svg)](https://icml.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2512.01334-b31b1b.svg)](https://arxiv.org/abs/2512.01334)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://law1223.github.io/AlignVid/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OmitI2V-ffce44.svg)](https://huggingface.co/datasets/AIPeanutman/OmitI2V)
[![License](https://img.shields.io/badge/License-Apache%202.0-3da639.svg)](./LICENSE)

**Training-free** · **model-agnostic** · **no fine-tuning** · **no extra weights** · **<0.1% overhead**

</div>

> **AlignVid** is a tiny, training-free attention-rebalancing recipe that makes
> generative models actually follow the prompt. Originally proposed for
> **text-guided image-to-video (TI2V)**, the *same* recipe — no architecture
> changes, no retraining — also transfers to **text-to-video (T2V)**,
> **text-to-image (T2I)**, and **image editing (I2I)**, with the same default
> coefficient γ = 1.35. Validated on **FramePack**, **FramePack-F1**, **Wan2.1**,
> and on VBench / GenEval / ImgEdit benchmarks.

<div align="center">

### 🌐 Works across tasks — one recipe, four task families

|  | **TI2V** | **T2V** | **T2I** | **I2I (editing)** |
|---|:---:|:---:|:---:|:---:|
| Backbones in the paper | FramePack · FramePack-F1 · Wan2.1 | Wan2.1 | OmniGen2 | OmniGen2 |
| Benchmark | **OmitI2V** *(this work)* | VBench | GenEval | ImgEdit |
| AlignVid effect | ✅ higher semantic fidelity | ✅ gain on T2V | ✅ gain on T2I | ✅ gain on editing |
| Runnable in this repo | ✅ | ✅ | 🛠️ adapt with ~few lines | 🛠️ adapt with ~few lines |

</div>

> The TI2V/T2V integrations (Wan2.1, FramePack, FramePack-F1) are shipped here as
> ready-to-run code. For T2I and image editing, the same ASM + GS recipe is just
> a few lines on the attention module of your model of choice — see
> [`models/Wan2.1/wan/modules/model.py`](models/Wan2.1/wan/modules/model.py) and
> [`models/FramePack/diffusers_helper/models/hunyuan_video_packed.py`](models/FramePack/diffusers_helper/models/hunyuan_video_packed.py)
> for reference implementations.

<div align="center">

📄 [Paper](https://arxiv.org/abs/2512.01334) &nbsp;·&nbsp; 🌐 [Project Page](https://law1223.github.io/AlignVid/) &nbsp;·&nbsp; 🤗 [OmitI2V Benchmark](https://huggingface.co/datasets/AIPeanutman/OmitI2V)

</div>

---

## ✨ Overview

Text-guided generative models — image-to-video, text-to-video, text-to-image,
image editing — share a common failure: when the prompt demands *substantial*
change, the model often **ignores it and just reproduces the input**. We trace
this to **visual dominance**: the visual condition causes severe attention
dispersion (high entropy) that suppresses the text signal, so the model tends to
*copy the input* instead of *following the prompt*. The failure mode is **not
TI2V-specific** — the same imbalance shows up in T2V, T2I, and image editing.

AlignVid re-calibrates attention internally, from an **energy-based
perspective**, with two components:

| Component | What it does |
|---|---|
| 🎯 **Attention Scaling Modulation (ASM)** | Rescales query/key representations by a single coefficient **γ** to sharpen the attention energy landscape — lowering entropy and amplifying text tokens over visual priors. |
| 🗓️ **Guidance Scheduling (GS)** | Applies ASM **selectively** across transformer blocks (foreground-sensitive) and denoising steps (early steps), stabilizing generation and limiting quality degradation. |

Unlike input-level perturbations such as blurring (which visibly corrupt the
input), AlignVid performs this reallocation **entirely within the model**,
giving a **tunable semantic–quality trade-off without input-level corruption**.
The default coefficient **γ = 1.35** transfers across both **backbones and task
families** without per-model search — think of it as a lightweight CFG-style
strength knob that works on whatever generative model you already have.

## 🔥 Highlights

- **Training-free** — a pure inference-time hook; no fine-tuning, no extra parameters.
- **Model- and task-agnostic** — the *same* recipe works on TI2V, T2V, T2I, and image editing, with the same default γ.
- **Negligible cost** — < 0.1% inference overhead.
- **Plug-and-play** — enabled through a CLI flag on the shipped TI2V/T2V backbones (FramePack / FramePack-F1 / Wan2.1); porting to a new model takes only a few lines on the attention module.
- **Benchmark included** — **OmitI2V**, 367 human-annotated add/delete/modify samples with VQA-style questions.

## 📁 Repository structure

```
.
├── models/
│   ├── Wan2.1/         # AlignVid integrated into Wan2.1 (I2V / T2V)
│   └── FramePack/      # AlignVid integrated into FramePack & FramePack-F1
├── evaluation/
│   ├── generation/     # batch video generation over the OmitI2V benchmark
│   ├── vqa_based/      # VQA-based semantic-fidelity evaluator (Qwen2.5-VL)
│   └── tool_based/     # tool-based VBench metrics (evaluate_i2v.py)
├── scripts/            # quick CLI: one video from your own image + prompt
├── assets/             # demo videos used on the project page
└── README.md
```

> **Note.** `models/Wan2.1` and `models/FramePack` are modified forks of the
> upstream projects. The AlignVid changes are the attention-scaling hooks (see
> [Usage](#-usage)) and the JSON-driven generation scripts. Base setup and model
> weights follow the upstream instructions linked in [Installation](#-installation).

## 🔧 Installation

AlignVid adds no new dependencies beyond the two backbones. Set up whichever
backbone you want to use, following its upstream instructions, then use the
scripts in this repo.

<details>
<summary><b>Wan2.1</b> — weights are downloaded manually</summary>

```bash
cd models/Wan2.1
pip install -r requirements.txt

# download the I2V-14B weights, then pass the folder via --ckpt_dir
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
```
Base repo: <https://github.com/Wan-Video/Wan2.1> · weights: [`Wan-AI/Wan2.1-I2V-14B-480P`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P).
Set `--ckpt_dir` (or `CKPT_DIR` in the scripts) to the downloaded folder.
</details>

<details>
<summary><b>FramePack / FramePack-F1</b> — weights download automatically</summary>

```bash
cd models/FramePack
pip install -r requirements.txt
```
No manual download needed: on first run the weights are fetched from Hugging Face
into `models/FramePack/hf_download/` (~30 GB) and cached. The repos used are
[`hunyuanvideo-community/HunyuanVideo`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo),
[`lllyasviel/flux_redux_bfl`](https://huggingface.co/lllyasviel/flux_redux_bfl),
and the transformer
([`FramePack_F1_I2V_HY_20250503`](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503)
for F1, [`FramePackI2V_HY`](https://huggingface.co/lllyasviel/FramePackI2V_HY) for v1).
Base repo: <https://github.com/lllyasviel/FramePack>.
</details>

**Summary:** Wan2.1 weights are downloaded manually and passed via `--ckpt_dir`;
FramePack weights download automatically on first run.

## 🚀 Usage

AlignVid is enabled purely through CLI flags — no retraining. The quickest start
is the helper scripts in [`scripts/`](./scripts): edit the `IMAGE` / `PROMPT`
variables at the top of one and run it to generate a video from **your own image
+ prompt** (output: `sample.mp4`).

```bash
bash scripts/run_framepack_f1.sh
```

The full flags (and how to run AlignVid over the whole OmitI2V benchmark, under
[`evaluation/generation/`](./evaluation/generation)) are documented below.

### Wan2.1 (Image-to-Video)

The recommended AlignVid setting scales the **image-side query** and **text-side
key** by **γ = 1.35** (everything else 1.0):

```bash
cd models/Wan2.1
python generate_i2v_json.py \
    --task i2v-14B \
    --size 480*832 \
    --ckpt_dir /path/to/model_weights/Wan2.1-I2V-14B-480P \
    --prompt_file /path/to/OmitI2V/meta.json \
    --output_dir ./results/Wan2.1 \
    --data_dir /path/to/data \
    --self_attn_scale 1.0 \
    --cross_attn_q_image_scale 1.35 \
    --cross_attn_k_image_scale 1.0 \
    --cross_attn_q_text_scale 1.0 \
    --cross_attn_k_text_scale 1.35 \
    --sample_guide_scale 1.0
```

| Flag | Meaning |
|---|---|
| `--cross_attn_q_image_scale` | ASM scale on image-side **query** (γ) |
| `--cross_attn_k_text_scale`  | ASM scale on text-side **key** (γ) |
| `--cross_attn_k_image_scale`, `--cross_attn_q_text_scale`, `--self_attn_scale` | other attention sites (default 1.0 = off) |
| `--sample_guide_scale` | sampler guidance scale |

> Setting all scales to `1.0` reproduces the unmodified Wan2.1 baseline.

### FramePack / FramePack-F1 (Image-to-Video)

For FramePack the AlignVid coefficient is a single `--control_scale` (γ):

```bash
cd models/FramePack
# FramePack-F1
python inference_f1_json.py \
    --prompt_file /path/to/OmitI2V/meta.json \
    --output_dir ./results/FramePack_F1 \
    --data_dir /path/to/data \
    --control_scale 1.35
# FramePack (v1): use inference_v1_json.py with the same flags
```

> `--control_scale 1.0` = baseline. The `--blur_img` flag reproduces the
> pilot-study input-blur baseline for comparison.

### Guidance Scheduling & variants (configuration)

The two Guidance Scheduling components and the ablation variants are controlled
by module-level switches at the top of each backbone's attention module:

- Wan2.1 → [`models/Wan2.1/wan/modules/model.py`](models/Wan2.1/wan/modules/model.py)
- FramePack → [`models/FramePack/diffusers_helper/models/hunyuan_video_packed.py`](models/FramePack/diffusers_helper/models/hunyuan_video_packed.py)

| Switch | Meaning | Default |
|---|---|---|
| `ALIGNVID_FOREGROUND_BLOCKS` | **BGS**: foreground-sensitive block indices (paper "Foreground-sensitive block indices" table). ASM is applied only on these blocks. | paper table |
| `ALIGNVID_VARIANT` *(FramePack only)* | preset to use: `framepack_f1` or `framepack` | `framepack_f1` |
| `ALIGNVID_STEP_WINDOW` | **SGS**: denoising-step window — `early` / `mid` / `late` / `all` (timesteps run 1000→0, so `early` = large t) | `early` |
| `ALIGNVID_BLUR_KEY` *(FramePack)* / `--blur_img` | pilot-study blur baseline | `False` |

ASM uses the paper's main **scalar scaling** (a fixed coefficient γ). The
defaults reproduce the paper's main setting; set `ALIGNVID_STEP_WINDOW='all'` or
swap block presets to reproduce the corresponding ablations. To disable AlignVid
entirely, set the scale flags to `1.0`.

> The foreground-sensitive block indices are shipped as the paper-reported
> presets. The procedure used to derive them (foreground-ratio analysis) is
> described in the paper appendix.

> **Single- vs multi-GPU.** AlignVid is applied identically on both paths: Wan2.1
> covers the standard forward and the xDiT/USP sequence-parallel forward
> (`wan/distributed/xdit_context_parallel.py`); FramePack covers the standard and
> sequence-parallel attention, with TeaCache on or off. The `flf2v` and `VACE`
> paths are outside the paper's scope and are not AlignVid-wired.

## 📦 OmitI2V benchmark

**OmitI2V** is our benchmark of **367 human-annotated samples** spanning
**modification / addition / deletion** scenarios, with VQA-style yes/no
questions for fine-grained edit compliance.

<div align="center">

🤗 **<https://huggingface.co/datasets/AIPeanutman/OmitI2V>**

</div>

Download it and point `--prompt_file` (to `meta.json`) and `--data_dir` (to the
dataset root) at the local copy.

## 📊 Evaluation

| Axis | Tool | Notes |
|---|---|---|
| **Semantic alignment** *(primary)* | `evaluation/vqa_based/` | VQA yes/no with Qwen2.5-VL-32B (per-sample macro accuracy) |
| **Visual quality** | `evaluation/tool_based/evaluate_i2v.py` | VBench-style metrics ([install VBench](https://github.com/Vchitect/VBench)) |

Convenience scripts (run each in its own env):

```bash
bash evaluation/eval_vqa.sh  <video_dir> <name>   # semantic fidelity (primary)
bash evaluation/eval_tool.sh <video_dir> <name>   # visual quality (Dynamic Degree + Aesthetic Quality)
```

VBench is **not vendored** — install it from upstream. See
[`evaluation/README.md`](./evaluation/README.md) for full details.

## 📈 Results

AlignVid yields consistent gains in **semantic fidelity** and **dynamic degree**
across FramePack, FramePack-F1, and Wan2.1, with only a minor drop in aesthetic
quality and **negligible (<0.1%) inference overhead**. See the paper for the full
tables.

To make setup verification easy, the table below compares the paper numbers with
an independent full-OmitI2V reproduction using this repository on FramePack-F1.
All numbers are percentages.

| Metric | FramePack-F1 baseline<br>(this repo / paper) | FramePack-F1 + AlignVid<br>(this repo / paper) | Gain<br>(this repo / paper) |
|---|---:|---:|---:|
| Modification | 66.25 / 64.45 | 72.08 / 71.27 | +5.83 / +6.82 |
| Addition | 69.72 / 67.79 | 73.53 / 71.60 | +3.81 / +3.81 |
| Deletion | 62.37 / 58.50 | 64.88 / 61.06 | +2.51 / +2.56 |
| Semantic Avg. | 66.11 / 63.58 | 70.16 / 67.98 | +4.05 / +4.40 |
| Dynamic Degree | 20.98 / 24.42 | 29.70 / 33.16 | +8.72 / +8.74 |
| Aesthetic Quality | 62.23 / 63.10 | 61.24 / 62.10 | -0.99 / -1.00 |
| Tool Avg. | 41.60 / 43.76 | 45.47 / 47.63 | +3.87 / +3.87 |

Slight absolute differences may arise from hardware, dependency, and video
preprocessing details. For VBench-style tool metrics, use separate temporary
folders for different runs (or run them sequentially), since sharing the same
resized-video cache can corrupt the comparison.

## 🎥 Demo

Side-by-side comparison clips (**baseline** vs. **AlignVid**) are in
[`assets/`](./assets), and play directly on the
**[project page](https://law1223.github.io/AlignVid/)**.

| Backbone | Baseline | AlignVid (ours) |
|---|---|---|
| FramePack | [`assets/Framepack/original/`](./assets/Framepack/original) | [`assets/Framepack/ours/`](./assets/Framepack/ours) |
| FramePack-F1 | [`assets/Framepack_f1/original/`](./assets/Framepack_f1/original) | [`assets/Framepack_f1/ours/`](./assets/Framepack_f1/ours) |
| Wan2.1 | [`assets/Wan2.1/original/`](./assets/Wan2.1/original) | [`assets/Wan2.1/ours/`](./assets/Wan2.1/ours) |

## 📝 Citation

If you find AlignVid useful, please consider citing:

```bibtex
@article{liu2025alignvid,
  title={AlignVid: Training-Free Attention Scaling for Semantic Fidelity in Text-Guided Image-to-Video Generation},
  author={Liu, Yexin and Shu, Wen-Jie and Huang, Zile and Zheng, Haoze and Wang, Yueze and Zhang, Manyuan and Lim, Ser-Nam and Yang, Harry},
  journal={arXiv preprint arXiv:2512.01334},
  year={2025}
}
```

## 🙏 Acknowledgements

AlignVid builds on these excellent open-source projects:
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
[FramePack](https://github.com/lllyasviel/FramePack), and
[VBench](https://github.com/Vchitect/VBench). Please also cite and follow the
licenses of these works.

## ⚖️ License

AlignVid's own code is released under the **Apache-2.0 License** (see
[LICENSE](./LICENSE) and [NOTICE](./NOTICE)). Code under `models/Wan2.1/` and
`models/FramePack/` derives from the respective upstream projects and remains
under their original licenses (see the `LICENSE`/`LICENSE.txt` files inside those
folders).

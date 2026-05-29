# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
from tqdm import tqdm
import random

import torch
import torch.distributed as dist

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Batch-generate Wan text-to-video samples from a JSON manifest.")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/path/to/VBench_full_info.json",
        help="Path to the JSON file that stores prompts.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/output_dir",
        help="Directory to save generated videos.")
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help=
        "The area (width*height) of the generated video. Should match WAN checkpoints."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help=
        "Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Override the output path for a single generation (debug only).")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt override (debug only).")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--self_attn_scale",
        type=float,
        default=1.0,
        help="Self-attention scale.")
    parser.add_argument(
        "--cross_attn_q_image_scale",
        type=float,
        default=1.0,
        help="Cross-attention q image scale.")
    parser.add_argument(
        "--cross_attn_k_image_scale",
        type=float,
        default=1.0,
        help="Cross-attention k image scale.")
    parser.add_argument(
        "--cross_attn_q_text_scale",
        type=float,
        default=1.0,
        help="Cross-attention q text scale.")
    parser.add_argument(
        "--cross_attn_k_text_scale",
        type=float,
        default=1.0,
        help="Cross-attention k text scale.")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _sanitize_prompt(prompt, fallback="sample"):
    sanitized = prompt.replace(" ", "_").replace("/", "_")
    sanitized = "".join(
        ch for ch in sanitized if ch.isalnum() or ch in ["_", "-", "."])
    sanitized = sanitized[:50]
    return sanitized if sanitized else fallback


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), "context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, (
            "The number of ulysses_size and ring_size should equal the world size."
        )
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model, is_vl=False, device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
    else:
        prompt_expander = None

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.prompt_file, "r", encoding="utf-8") as file:
        samples = json.load(file)

    for idx, sample in enumerate(
            tqdm(samples, desc="Generating text-to-video samples")):
        prompt = sample.get("prompt_en") or sample.get("prompt") or args.prompt
        if prompt is None:
            logging.warning(f"Sample {idx} missing prompt, skipping.")
            continue

        effective_prompt = prompt
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    effective_prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status is False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = effective_prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            effective_prompt = input_prompt[0]
            logging.info(f"Extended prompt: {effective_prompt}")

        prompt_suffix = _sanitize_prompt(effective_prompt,
                                         fallback=f"sample_{idx:05d}")
        sample_id = f"{prompt_suffix}-0"
        save_path = args.save_file or os.path.join(args.output_dir,
                                                   f"{sample_id}.mp4")

        if os.path.exists(save_path):
            logging.info(f"Skipping {sample_id} because it already exists.")
            continue

        logging.info(f"Input prompt: {effective_prompt}")

        video = wan_t2v.generate(
            effective_prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            self_attn_scale=args.self_attn_scale,
            cross_attn_q_image_scale=args.cross_attn_q_image_scale,
            cross_attn_k_image_scale=args.cross_attn_k_image_scale,
            cross_attn_q_text_scale=args.cross_attn_q_text_scale,
            cross_attn_k_text_scale=args.cross_attn_k_text_scale)

        if rank == 0:
            logging.info(f"Saving generated video to {save_path}")
            cache_video(
                tensor=video[None],
                save_file=save_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)

'''
python vbench_t2i_json.py \
    --task t2v-14B \
    --size 480*832 \
    --ckpt_dir /path/to/model_weights/Wan2.1-T2V-14B-480P \
    --prompt_file /path/to/project/OmitT2V/meta_v3.json \
    --output_dir /path/to/project/OmitT2V/Results/Wan2_1 
'''
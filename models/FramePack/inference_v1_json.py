import os
import argparse
import torch
import traceback
import einops
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
import torch.distributed as dist

import imageio
# diffusers and transformers imports
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

# helper modules
from diffusers_helper.hf_login import login
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
# from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.utils import crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from diffusers_helper.parallel_mgr import init_group, init_parallel_env
# from diffusers_helper.memory import updata_gpu
# set Hugging Face cache directory
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# from diffusers_helper.vae_parallel import parallel_vae_tile
def save_bcthw_as_mp4(x, output_filename, fps=10):
    b, c, t, h, w = x.shape

    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break

    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    
    # Normalize and convert to uint8
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    
    # Rearrange tensor to video format
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    
    # Convert to numpy array and clip values
    frames_np = x.numpy().clip(0, 255).astype(np.uint8)

    # Create the video with imageio
    with imageio.get_writer(output_filename, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames_np:
            writer.append_data(frame)
    
    print(f"Saved video: {output_filename}")
    return x



def preprocess_image_for_i2v(pil_img, target_size=512, blur_radius=1):
    w, h = pil_img.size
    if w > target_size and h > target_size:
        scale = target_size / min(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return np.array(pil_img)

def main(args):  
    # print("-------------",dist.get_rank(), gpu)
    # print( "-------------",dist.get_rank(), torch.cuda.current_device())

    # init_parallel_env()
    # init_group()
    # global gpu
    gpu = torch.device(f'cuda:{torch.cuda.current_device()}')

    # check available GPU memory
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    print(f'Free VRAM: {free_mem_gb} GB, High-VRAM Mode: {high_vram}')

    # load models
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

    # set eval mode and dtype
    for m in [vae, text_encoder, text_encoder_2, image_encoder, transformer]:
        m.eval()

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    for m in [vae, text_encoder, text_encoder_2, image_encoder, transformer]:
        m.requires_grad_(False)

    # device allocation
    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)
    if True:
        vae.enable_slicing()
        vae.enable_tiling()
        # parallel_vae_tile(vae, docode_out_name="sample")
    # create outputs folder
    def ensure_folder(path):
        os.makedirs(path, exist_ok=True)

    output_dir = args.output_dir
    ensure_folder(output_dir)

    @torch.no_grad()
    def worker(input_image, sample_id, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, control_scale, warmup=False):
        """
        Core generation worker: processes input image and generates a video.
        """
        file_base_name = sample_id
        filename = os.path.join(output_dir, f"{file_base_name}.mp4")

        total_latent_sections = int(max(round((total_second_length * 30) / (latent_window_size * 4)), 1))
        job_id = generate_timestamp()

        # Clean GPU models if using low VRAM
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        # Encode prompts
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Process input image
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision encoding
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(transformer.dtype)

        # Move text and VAE latents to transformer dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)

        # Sampling
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros((1, 16, 1+2+16, height//8, width//8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = ([3] + [2]*(total_latent_sections-3) + [1,0]) if total_latent_sections>4 else list(reversed(range(total_latent_sections)))


        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size


            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def cb(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                
                return preview[-1]

            generated = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=cb,
                control_scale=control_scale
            )

            if is_last_section:
                generated = torch.cat([start_latent.to(generated), generated], dim=2)

            total_generated_latent_frames += generated.shape[2]
            history_latents = torch.cat([generated.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_lat = history_latents[:, :, :total_generated_latent_frames]
            if history_pixels is None:
                history_pixels = vae_decode(real_lat, vae).cpu()
            else:
                curr_pixels = vae_decode(real_lat[:, :, :latent_window_size*2 + (1 if is_last_section else 0)], vae).cpu()
                history_pixels = soft_append_bcthw(curr_pixels, history_pixels, latent_window_size*4 - 3)

            if is_last_section and not warmup:
                if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                    save_bcthw_as_mp4(history_pixels, filename, fps=30)
                    print(f"Saved video: {filename}")
                break

        return
        
    import time
    import json
    torch.cuda.synchronize()
    start_time=time.time()
    with open(args.prompt_file, 'r') as f:
        samples = json.load(f)
    
    for sample in samples:
        start_time = time.time()
        pil_img = Image.open(os.path.join(args.data_dir, sample['image-path'])).convert("RGB")
        if args.blur_img:
            img = preprocess_image_for_i2v(pil_img)
        else:
            img = np.array(pil_img)
        
        prompt = sample['prompt']
        sample_id = sample['id']
        if os.path.exists(os.path.join(output_dir, f"{sample_id}.mp4")):
            print(f"Skipping {sample_id} because it already exists")
            continue
        else:
            worker(
                img,
                sample_id,
                prompt,
                args.n_prompt,
                args.seed,
                args.length,
                args.window,
                args.steps,
                args.cfg,
                args.gs,
                args.rs,
                args.gpu_mem,
                args.teacache,
                args.control_scale,
                warmup=False
            )
        print("use time", time.time() - start_time)
    torch.cuda.synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FramePack CLI for video generation')
    parser.add_argument('--prompt_file', type=str, default="/path/to/prompt_file.json", help='Path to prompt file')
    parser.add_argument('--output_dir', type=str, default="/path/to/output_dir", help='Path to output directory')
    parser.add_argument('--data_dir', type=str, default="/path/to/data_dir", help='Path to data directory')
    parser.add_argument('--n_prompt', type=str, default='', help='Negative prompt text')
    parser.add_argument('--seed', type=int, default=31337, help='Random seed')
    parser.add_argument('--length', type=float, default=6.0, help='Total video length in seconds')
    parser.add_argument('--window', type=int, default=9, help='Latent window size')
    parser.add_argument('--steps', type=int, default=25, help='Number of inference steps')
    parser.add_argument('--cfg', type=float, default=1.0, help='CFG scale')
    parser.add_argument('--gs', type=float, default=10.0, help='Distilled CFG scale')
    parser.add_argument('--rs', type=float, default=0.0, help='CFG rescale')
    parser.add_argument('--gpu_mem', type=float, default=79.0, help='GPU memory preservation (GB)')
    parser.add_argument('--teacache', default=True, help='Use TeaCache for speed')
    parser.add_argument('--blur_img', action='store_true', help='Apply Gaussian blur to input image')
    parser.add_argument('--control_scale', type=float, default=1.0, help='Control scale')
    args = parser.parse_args()

    init_parallel_env()
    init_group()
    main(args)  

'''
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --nnodes=1 --master_port=39502 inference_v1_json.py --prompt_file /path/to/project/OmitI2V/meta_v3.json --output_dir /path/to/project/FramePack/outputs --data_dir /path/to/project --control_scale 1.0

torchrun --nproc-per-node=1 --nnodes=1 --master_port=39504 inference_v1_json.py \
  --control_scale 1.0 \
  --data_dir /path/to/project \
  --prompt_file /path/to/project/OmitI2V/meta_test.json \
  --output_dir /path/to/project/OmitI2V_Results/videos/FramePack_v1/time_o
'''
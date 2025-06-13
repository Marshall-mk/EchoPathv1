import argparse
import logging
import math
import os
import shutil
import json
from glob import glob
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from packaging import version
from functools import partial
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from echo.common.datasets import TensorSet, ImageSet, TensorSetv3, TensorSetv4
from echo.common import (
    pad_reshape,
    unpad_reshape,
    padf,
    unpadf,
    load_model,
    save_as_mp4,
    save_as_gif,
    save_as_img,
    save_as_avi,
    parse_formats,
    FlowMatchingScheduler,
)

"""
CUDA_VISIBLE_DEVICES='2' python -m echo.lvdm.sample  
	--config echo/lvdm/configs/cardiacnet.yaml   
	--unet /nfs/usrhome/khmuhammad/Echodream/experiments/lvdm_cardiacnet/checkpoint-100000/unet_ema   
	--vae /nfs/usrhome/khmuhammad/Echodream/models/vae   
	--conditioning /nfs/usrhome/khmuhammad/Echodream/data/latents/cardiacnet/Latents   
	--output /nfs/usrhome/khmuhammad/Echodream/samples/lvdm_cardiacnet  
	--num_samples 655    
	--batch_size 48    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type class_id 
	--class_ids 4
    --condition_guidance_scale 5.0
    --frame_guidance_scale 1.0
    --use_separate_guidance
    --seed 42
"""


def tokenize_text(text, tokenizer):
    """Tokenizes the input text using the provided tokenizer"""
    tokenized_text = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return tokenized_text.input_ids, tokenized_text.attention_mask


def get_conditioning_vector(
    conditioning_type, conditioning_value, B, device, dtype, tokenizer, text_encoder
):
    """
    Create conditioning vectors based on the specified type
    """
    if conditioning_type == "class_id":
        # Integer class IDs
        if isinstance(conditioning_value, int):
            # Random class IDs up to conditioning_value
            cond = torch.randint(
                0, conditioning_value, (B,), device=device, dtype=dtype
            )
        else:
            # Fixed class ID
            cond = torch.tensor(
                [int(conditioning_value)] * B, device=device, dtype=dtype
            )

        # Format for model: B -> B x 1 x 1
        return cond[:, None, None]

    elif conditioning_type == "lvef":
        # LVEF values (usually between 0-100)
        if (
            isinstance(conditioning_value, (list, tuple))
            and len(conditioning_value) == 2
        ):
            # Random LVEF in range
            min_val, max_val = conditioning_value
            cond = (
                torch.rand(B, device=device, dtype=dtype) * (max_val - min_val)
                + min_val
            )
        else:
            # Fixed LVEF value
            cond = torch.tensor(
                [float(conditioning_value)] * B, device=device, dtype=dtype
            )

        # Format for model: B -> B x 1 x 1
        return cond[:, None, None]

    elif conditioning_type == "view":
        # View type as integer ID
        if isinstance(conditioning_value, int):
            # Random view IDs up to conditioning_value
            cond = torch.randint(
                0, conditioning_value, (B,), device=device, dtype=dtype
            )
        else:
            # Fixed view ID
            cond = torch.tensor(
                [int(conditioning_value)] * B, device=device, dtype=dtype
            )

        # Format for model: B -> B x 1 x 1
        return cond[:, None, None]

    elif conditioning_type == "text":
        text_templates = [
            "An echocardiography video with {} condition",
            "An echocardiography video with Left Ventricular Ejection Fraction {}%",
        ]
        class_names = [
            "Atrial Septal Defect",
            "Non-Atrial Septal Defect",
            "Non-Pulmonary Arterial Hypertension",
            "Pulmonary Arterial Hypertension",
        ]

        # Generate random text conditioning
        text_conditioning = []
        for _ in range(B // 2):
            # Randomly combine text_templates[0] with class names
            random_class = class_names[torch.randint(0, len(class_names), (1,)).item()]
            text_conditioning.append(text_templates[0].format(random_class))

        for _ in range(B // 2):
            # Randomly combine text_templates[1] with a random LVEF value
            random_lvef = torch.randint(10, 90, (1,)).item()
            text_conditioning.append(text_templates[1].format(random_lvef))

        # Tokenize the text conditioning
        input_ids, attention_mask = tokenize_text(text_conditioning, tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        conditioning = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.to(dtype=dtype)

        return conditioning

    else:
        raise ValueError(f"Unsupported conditioning type: {conditioning_type}")


def load_text_encoder_and_tokenizer(text_encoder_path, tokenizer_path=None):
    """
    Loads the text encoder and tokenizer for text-conditioned generation.

    Args:
        text_encoder_path: Path to the text encoder model or pretrained model name
        tokenizer_path: Path to the tokenizer. If None, will use text_encoder_path

    Returns:
        tuple: (text_encoder, tokenizer) - Loaded models for inference
    """
    if tokenizer_path is None:
        tokenizer_path = text_encoder_path

    # Check if the paths are local directories or HF model IDs
    if os.path.isdir(text_encoder_path):
        # Load from local directory
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
    else:
        # Load from Hugging Face
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)

    if os.path.isdir(tokenizer_path):
        # Load from local directory
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    else:
        # Load from Hugging Face
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    print(f"Loaded text encoder from {text_encoder_path}")
    print(f"Loaded tokenizer from {tokenizer_path}")

    return text_encoder, tokenizer


if __name__ == "__main__":
    # 1 - Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--unet", type=str, default=None, help="Path unet checkpoint.")
    parser.add_argument("--vae", type=str, default=None, help="Path vae checkpoint.")
    parser.add_argument(
        "--conditioning",
        type=str,
        default=None,
        help="Path to the folder containing the conditioning latents/images.",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory.")
    parser.add_argument(
        "--num_samples", type=int, default=8, help="Number of samples to generate."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps.")

    # New arguments for flexible conditioning and sampling
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="diffusion",
        choices=["diffusion", "flow_matching"],
        help="Sampling method to use.",
    )
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="class_id",
        choices=["class_id", "lvef", "view", "text", "csv"],
        help="Type of conditioning to use.",
    )

    # Conditioning value arguments - one will be used based on conditioning_type
    parser.add_argument(
        "--class_ids",
        type=int,
        default=3,
        help="Number of class ids or specific class id.",
    )
    parser.add_argument(
        "--lvef_range",
        type=float,
        nargs=2,
        default=[10, 90],
        help="Min and max LVEF values.",
    )
    parser.add_argument("--lvef", type=float, default=None, help="Specific LVEF value.")
    parser.add_argument(
        "--view_ids",
        type=int,
        default=4,
        help="Number of view ids or specific view id.",
    )

    parser.add_argument(
        "--save_as",
        type=parse_formats,
        default=None,
        help="Save formats separated by commas (e.g., avi,jpg). Available: avi, mp4, gif, jpg, png, pt",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=192,
        help="Number of frames to generate. Must be a multiple of 32",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--condition_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for class conditioning (1.0=no guidance).",
    )
    parser.add_argument(
        "--frame_guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for frame conditioning (1.0=no guidance).",
    )
    parser.add_argument(
        "--use_separate_guidance",
        action="store_true",
        help="Use separate guidance scales for class and frame conditioning.",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use DDIM sampler (only for diffusion mode).",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Path to text encoder model (required for text conditioning).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (if different from text encoder path).",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # 2 - Load models
    unet = load_model(args.unet)
    vae = load_model(args.vae)

    # Load text encoder and tokenizer if using text conditioning
    text_encoder = None
    tokenizer = None
    if args.conditioning_type == "text":
        if args.text_encoder is None:
            raise ValueError("Text encoder path must be provided for text conditioning")
        text_encoder, tokenizer = load_text_encoder_and_tokenizer(
            args.text_encoder, args.tokenizer
        )

    # 3 - Load or create scheduler based on sampling_mode
    if args.sampling_mode == "diffusion":
        scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler)
        scheduler_klass_name = scheduler_kwargs.pop("_class_name")
        if args.ddim:
            print("Using DDIMScheduler")
            scheduler_klass_name = "DDIMScheduler"
            scheduler_kwargs.pop("variance_type", None)
        scheduler_klass = getattr(diffusers, scheduler_klass_name, None)
        assert scheduler_klass is not None, (
            f"Could not find scheduler class {scheduler_klass_name}"
        )
        scheduler = scheduler_klass(**scheduler_kwargs)
    else:  # flow_matching
        print("Using FlowMatchingScheduler")
        scheduler = FlowMatchingScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 1000)
        )

    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps

    # 4 - Load dataset for reference frames
    ## detect type of conditioning:
    file_ext = os.listdir(args.conditioning)[0].split(".")[-1].lower()
    assert file_ext in ["pt", "jpg", "png"], (
        f"Conditioning files must be either .pt, .jpg or .png, not {file_ext}"
    )
    if file_ext == "pt":
        # dataset = TensorSetv3(args.conditioning, num_frames=1, split=["TEST"])
        dataset = TensorSetv4(args.conditioning)
    else:
        dataset = ImageSet(args.conditioning, ext=file_ext)
    assert len(dataset) > 0, (
        f"No files found in {args.conditioning} with extension {file_ext}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # 5 - Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(
        args.seed
        if args.seed is not None
        else config.seed
        if hasattr(config, "seed")
        else np.random.randint(0, 1000000)
    )
    unet = unet.to(device, dtype)
    vae = vae.to(device, torch.float32)
    unet.eval()
    vae.eval()
    if text_encoder is not None:
        text_encoder = text_encoder.to(device)
        tokenizer = tokenizer.to(device)
        text_encoder.eval()

    format_input = (
        pad_reshape
        if config.unet._class_name == "UNetSpatioTemporalConditionModel"
        else padf
    )
    format_output = (
        unpad_reshape
        if config.unet._class_name == "UNetSpatioTemporalConditionModel"
        else unpadf
    )

    B, C, T, H, W = (
        args.batch_size,
        config.unet.out_channels,
        config.unet.num_frames,
        config.unet.sample_size,
        config.unet.sample_size,
    )
    fps = config.globals.target_fps if hasattr(config.globals, "target_fps") else 30

    # Stitching parameters
    args.frames = int(np.ceil(args.frames / 32) * 32)
    if args.frames > T:
        OT = T // 2  # overlap 64//2
        TR = (args.frames - T) / 32  # total frames (192 - 64) / 32 = 4
        TR = int(TR + 1)  # total repetitions
        NT = (T - OT) * TR + OT  # = args.frame
    else:
        OT = 0
        TR = 1
        NT = T

    # Forward kwargs setup
    forward_kwargs = {
        "timestep": -1,  # Will be updated in the loop
    }

    # Set up conditioning based on type
    if args.conditioning_type == "lvef" and args.lvef is not None:
        conditioning_value = args.lvef
    elif args.conditioning_type == "lvef":
        conditioning_value = args.lvef_range
    elif args.conditioning_type == "class_id":
        conditioning_value = args.class_ids
    elif args.conditioning_type == "view":
        conditioning_value = args.view_ids
    else:
        conditioning_value = None  # For text, and csv we'll handle differently

    if config.unet._class_name == "UNetSpatioTemporalConditionModel":
        dummy_added_time_ids = torch.zeros(
            (B * TR, config.unet.addition_time_embed_dim), device=device, dtype=dtype
        )
        forward_kwargs["added_time_ids"] = dummy_added_time_ids

    sample_index = 0
    filelist = []

    os.makedirs(args.output, exist_ok=True)
    for ext in args.save_as:
        os.makedirs(os.path.join(args.output, ext), exist_ok=True)
    finished = False

    pbar = tqdm(total=args.num_samples)

    # 6 - Generate samples
    with torch.no_grad():
        while not finished:
            for cond, value in dataloader:
                if finished:
                    break

                # Prepare latent noise
                latents = torch.randn(
                    (B, C, NT, H, W), device=device, dtype=dtype, generator=generator
                )

                # Get conditioning based on specified type
                if args.conditioning_type == "text":
                    conditioning, text_conditioning = get_conditioning_vector(
                        args.conditioning_type,
                        conditioning_value,
                        B,
                        device,
                        dtype,
                        tokenizer,
                        text_encoder,
                    )
                elif args.conditioning_type == "csv":
                    print("Loading conditioning from CSV")
                    conditioning = value[:, None, None]
                    conditioning = conditioning.to(device, dtype=dtype)
                else:
                    conditioning = get_conditioning_vector(
                        args.conditioning_type,
                        conditioning_value,
                        B,
                        device,
                        dtype,
                        tokenizer,
                        text_encoder,
                    )

                # Repeat conditioning for temporal stitching if needed
                conditioning = (
                    conditioning.repeat_interleave(TR, dim=0)
                    if TR > 1
                    else conditioning
                )

                # Set the correct keyword argument based on conditioning type
                forward_kwargs["encoder_hidden_states"] = conditioning

                # Prepare reference frames
                latent_cond_images = cond.to(device, torch.float32)
                if file_ext != "pt":
                    # project image to latent space
                    latent_cond_images = vae.encode(
                        latent_cond_images
                    ).latent_dist.sample()
                    latent_cond_images = latent_cond_images * vae.config.scaling_factor
                # this is strictly because we are using the video latents for conditioning and when loading
                # with tensorsetv2, the shape is (B, C, C, H, W) and we need to squeeze one of the C dimension
                # to get the shape (B, C, H, W)
                latent_cond_images = latent_cond_images.squeeze(1)
                # print(f"Latent conditioning shape: {latent_cond_images.shape}")
                latent_cond_images = latent_cond_images[:, :, None, :, :].repeat(
                    1, 1, NT, 1, 1
                )  # B x C x T x H x W

                # Apply classifier-free guidance if specified
                use_condition_guidance = args.condition_guidance_scale > 1.0
                use_frame_guidance = args.frame_guidance_scale > 1.0
                use_separate_guidance = args.use_separate_guidance

                # Denoise the latent
                with torch.autocast("cuda"):
                    for t in timesteps:
                        forward_kwargs["timestep"] = t

                        # Prepare model input
                        latent_model_input = scheduler.scale_model_input(
                            latents, timestep=t
                        )

                        # Split latent into noise part and conditioning part for frame guidance
                        latent_noise = latent_model_input
                        latent_cond = latent_cond_images
                        latent_model_input = torch.cat(
                            (latent_noise, latent_cond), dim=1
                        )  # B x 2C x T x H x W

                        # Format input for model
                        latent_model_input, padding = format_input(
                            latent_model_input, mult=3
                        )

                        if (
                            use_condition_guidance or use_frame_guidance
                        ) and args.sampling_mode == "diffusion":
                            # Create input combinations based on guidance settings
                            if (
                                use_separate_guidance
                                and use_condition_guidance
                                and use_frame_guidance
                            ):
                                # Four combinations: (class+frame, class-only, frame-only, none)

                                # 1. Fully conditional (class+frame)
                                full_cond_kwargs = forward_kwargs.copy()

                                # 2. Class-only (zero frame conditioning)
                                class_only_kwargs = forward_kwargs.copy()
                                class_only_input = latent_model_input.clone()
                                # Zero out frame conditioning part (second half of channels)
                                class_only_input[:, C:, :, :, :] = torch.zeros_like(
                                    class_only_input[:, C:, :, :, :]
                                )

                                # 3. Frame-only (zero class conditioning)
                                frame_only_kwargs = forward_kwargs.copy()
                                frame_only_kwargs["encoder_hidden_states"] = (
                                    torch.zeros_like(conditioning)
                                )

                                # 4. Unconditional (no class, no frame)
                                uncond_kwargs = forward_kwargs.copy()
                                uncond_kwargs["encoder_hidden_states"] = (
                                    torch.zeros_like(conditioning)
                                )
                                uncond_input = latent_model_input.clone()
                                uncond_input[:, C:, :, :, :] = torch.zeros_like(
                                    uncond_input[:, C:, :, :, :]
                                )

                                # Process each combination for stitching
                                inputs_list = []
                                for r in range(TR):
                                    inputs_list.append(
                                        latent_model_input[
                                            :, r * (T - OT) : r * (T - OT) + T
                                        ]
                                    )
                                inputs = torch.cat(inputs_list, dim=0)

                                # Run predictions
                                noise_pred_full = unet(
                                    inputs, **full_cond_kwargs
                                ).sample

                                inputs_class_only_list = []
                                for r in range(TR):
                                    inputs_class_only_list.append(
                                        class_only_input[
                                            :, r * (T - OT) : r * (T - OT) + T
                                        ]
                                    )
                                inputs_class_only = torch.cat(
                                    inputs_class_only_list, dim=0
                                )
                                noise_pred_class = unet(
                                    inputs_class_only, **full_cond_kwargs
                                ).sample

                                noise_pred_frame = unet(
                                    inputs, **frame_only_kwargs
                                ).sample

                                inputs_uncond_list = []
                                for r in range(TR):
                                    inputs_uncond_list.append(
                                        uncond_input[:, r * (T - OT) : r * (T - OT) + T]
                                    )
                                inputs_uncond = torch.cat(inputs_uncond_list, dim=0)
                                noise_pred_uncond = unet(
                                    inputs_uncond, **uncond_kwargs
                                ).sample

                                # Split and stitch predictions
                                outputs_full = torch.chunk(noise_pred_full, TR, dim=0)
                                outputs_class = torch.chunk(noise_pred_class, TR, dim=0)
                                outputs_frame = torch.chunk(noise_pred_frame, TR, dim=0)
                                outputs_uncond = torch.chunk(
                                    noise_pred_uncond, TR, dim=0
                                )

                                # Apply separate guidance scales
                                noise_predictions = []
                                for r in range(TR):
                                    full_chunk = (
                                        outputs_full[r]
                                        if r == 0
                                        else outputs_full[r][:, OT:]
                                    )
                                    class_chunk = (
                                        outputs_class[r]
                                        if r == 0
                                        else outputs_class[r][:, OT:]
                                    )
                                    frame_chunk = (
                                        outputs_frame[r]
                                        if r == 0
                                        else outputs_frame[r][:, OT:]
                                    )
                                    uncond_chunk = (
                                        outputs_uncond[r]
                                        if r == 0
                                        else outputs_uncond[r][:, OT:]
                                    )

                                    # Combined guidance: uncond + class_guidance*(class-uncond) + frame_guidance*(frame-uncond)
                                    guided_chunk = (
                                        uncond_chunk
                                        + args.condition_guidance_scale
                                        * (class_chunk - uncond_chunk)
                                        + args.frame_guidance_scale
                                        * (frame_chunk - uncond_chunk)
                                    )
                                    noise_predictions.append(guided_chunk)

                            else:
                                # Simplified guidance with single scale
                                # Create unconditional input
                                uncond_kwargs = forward_kwargs.copy()
                                uncond_kwargs["encoder_hidden_states"] = (
                                    torch.zeros_like(conditioning)
                                )

                                # Create unconditional input for frame guidance if needed
                                if use_frame_guidance:
                                    uncond_input = latent_model_input.clone()
                                    uncond_input[:, C:, :, :, :] = torch.zeros_like(
                                        uncond_input[:, C:, :, :, :]
                                    )
                                else:
                                    uncond_input = latent_model_input

                                # Stitching for conditional prediction
                                inputs_cond_list = []
                                for r in range(TR):
                                    inputs_cond_list.append(
                                        latent_model_input[
                                            :, r * (T - OT) : r * (T - OT) + T
                                        ]
                                    )
                                inputs_cond = torch.cat(inputs_cond_list, dim=0)
                                noise_pred_cond = unet(
                                    inputs_cond, **forward_kwargs
                                ).sample

                                # Stitching for unconditional prediction
                                inputs_uncond_list = []
                                for r in range(TR):
                                    inputs_uncond_list.append(
                                        uncond_input[:, r * (T - OT) : r * (T - OT) + T]
                                    )
                                inputs_uncond = torch.cat(inputs_uncond_list, dim=0)
                                noise_pred_uncond = unet(
                                    inputs_uncond, **uncond_kwargs
                                ).sample

                                # Apply guidance
                                outputs_cond = torch.chunk(noise_pred_cond, TR, dim=0)
                                outputs_uncond = torch.chunk(
                                    noise_pred_uncond, TR, dim=0
                                )

                                guidance_scale = (
                                    args.condition_guidance_scale
                                    if use_condition_guidance
                                    else args.frame_guidance_scale
                                )

                                noise_predictions = []
                                for r in range(TR):
                                    cond_chunk = (
                                        outputs_cond[r]
                                        if r == 0
                                        else outputs_cond[r][:, OT:]
                                    )
                                    uncond_chunk = (
                                        outputs_uncond[r]
                                        if r == 0
                                        else outputs_uncond[r][:, OT:]
                                    )
                                    guided_chunk = uncond_chunk + guidance_scale * (
                                        cond_chunk - uncond_chunk
                                    )
                                    noise_predictions.append(guided_chunk)
                        else:
                            # Standard prediction without guidance
                            # Stitching
                            inputs = torch.cat(
                                [
                                    latent_model_input[
                                        :, r * (T - OT) : r * (T - OT) + T
                                    ]
                                    for r in range(TR)
                                ],
                                dim=0,
                            )
                            noise_pred = unet(inputs, **forward_kwargs).sample
                            outputs = torch.chunk(noise_pred, TR, dim=0)

                            noise_predictions = []
                            for r in range(TR):
                                noise_predictions.append(
                                    outputs[r] if r == 0 else outputs[r][:, OT:]
                                )

                        # Combine noise predictions and format output
                        noise_pred = torch.cat(noise_predictions, dim=1)
                        noise_pred = format_output(noise_pred, pad=padding)

                        # Update latents based on sampling mode
                        if args.sampling_mode == "diffusion":
                            latents = scheduler.step(noise_pred, t, latents).prev_sample
                        else:  # flow_matching
                            # Euler step: x_t+1 = x_t - v(x_t, t) * dt
                            dt = 1.0 / (len(timesteps) - 1)
                            latents = latents - noise_pred * dt

                # VAE decode
                latents = rearrange(latents, "b c t h w -> (b t) c h w").cpu()
                latents = latents / vae.config.scaling_factor

                # Decode in chunks to save memory
                chunked_latents = torch.split(latents, args.batch_size, dim=0)
                decoded_chunks = []
                for chunk in chunked_latents:
                    decoded_chunks.append(vae.decode(chunk.float().cuda()).sample.cpu())
                video = torch.cat(decoded_chunks, dim=0)  # (B*T) x H x W x C

                # format output
                video = rearrange(video, "(b t) c h w -> b t h w c", b=B)
                video = (video + 1) * 128
                video = video.clamp(0, 255).to(torch.uint8)

                print(
                    f"Generated videos: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]"
                )

                # Get conditioning values for metadata
                if args.conditioning_type == "class_id":
                    cond_values = conditioning.squeeze()[::TR].to(torch.int).tolist()
                elif args.conditioning_type == "lvef":
                    cond_values = conditioning.squeeze()[::TR].tolist()
                elif args.conditioning_type == "view":
                    cond_values = conditioning.squeeze()[::TR].to(torch.int).tolist()
                elif args.conditioning_type == "csv":
                    cond_values = conditioning.squeeze()[::TR].to(torch.int).tolist()
                else:  # text
                    cond_values = text_conditioning

                # save samples
                for j in range(B):
                    # FileName,CondType,CondValue,FrameHeight,FrameWidth,FPS,NumberOfFrames,Split
                    filelist.append(
                        [
                            f"sample_{sample_index:06d}",
                            args.conditioning_type,
                            cond_values[j],
                            video.shape[2],
                            video.shape[3],
                            fps,
                            video.shape[1],
                            "GENERATED",
                        ]
                    )

                    # Save in requested formats
                    if "mp4" in args.save_as:
                        save_as_mp4(
                            video[j],
                            os.path.join(
                                args.output, "mp4", f"sample_{sample_index:06d}.mp4"
                            ),
                        )
                    if "avi" in args.save_as:
                        save_as_avi(
                            video[j],
                            os.path.join(
                                args.output, "avi", f"sample_{sample_index:06d}.avi"
                            ),
                        )
                    if "gif" in args.save_as:
                        save_as_gif(
                            video[j],
                            os.path.join(
                                args.output, "gif", f"sample_{sample_index:06d}.gif"
                            ),
                        )
                    if "jpg" in args.save_as:
                        save_as_img(
                            video[j],
                            os.path.join(
                                args.output, "jpg", f"sample_{sample_index:06d}"
                            ),
                            ext="jpg",
                        )
                    if "png" in args.save_as:
                        save_as_img(
                            video[j],
                            os.path.join(
                                args.output, "png", f"sample_{sample_index:06d}"
                            ),
                            ext="png",
                        )
                    if "pt" in args.save_as:
                        torch.save(
                            video[j].clone(),
                            os.path.join(
                                args.output, "pt", f"sample_{sample_index:06d}.pt"
                            ),
                        )

                    sample_index += 1
                    pbar.update(1)
                    if sample_index >= args.num_samples:
                        finished = True
                        break

    # Save metadata
    df = pd.DataFrame(
        filelist,
        columns=[
            "FileName",
            "CondType",
            "CondValue",
            "FrameHeight",
            "FrameWidth",
            "FPS",
            "NumberOfFrames",
            "Split",
        ],
    )
    df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)

    # Save generation parameters
    params = {
        "sampling_mode": args.sampling_mode,
        "conditioning_type": args.conditioning_type,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "condition_guidance_scale": args.condition_guidance_scale,
        "frame_guidance_scale": args.frame_guidance_scale,
        "use_separate_guidance": args.use_separate_guidance,
        "seed": args.seed,
        "frames": args.frames,
    }
    with open(os.path.join(args.output, "generation_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(
        f"Generated {sample_index} samples using {args.sampling_mode} with {args.conditioning_type} conditioning."
    )

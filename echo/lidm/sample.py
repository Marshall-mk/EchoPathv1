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
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from echo.common.datasets import instantiate_dataset
from echo.common import padf, unpadf, load_model, FlowMatchingScheduler

"""
CUDA_VISIBLE_DEVICES='2' python -m echo.lidm.sample  
	--config echo/lidm/configs/cardiacnet.yaml   
	--unet /nfs/usrhome/khmuhammad/Echodream/experiments/lidm_cardiacnet/checkpoint-4800/unet_ema   
	--vae /nfs/usrhome/khmuhammad/Echodream/models/vae   
	--output /nfs/usrhome/khmuhammad/Echodream/samples/lidm_cardiacnet  
	--num_samples 2000    
	--batch_size 246    
	--num_steps 256     
	--save_latent   
	--sampling_mode diffusion 
	--conditioning_type class_id 
	--class_ids 4
    --condition_guidance_scale 5.0
    --seed 0

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
        choices=["class_id", "view", "text"],
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
        "--view_ids",
        type=int,
        default=4,
        help="Number of view ids or specific view id.",
    )
    parser.add_argument(
        "--condition_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for  conditioning (1.0=no guidance).",
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
    parser.add_argument("--output", type=str, default=".", help="Output directory.")
    parser.add_argument(
        "--num_samples", type=int, default=8, help="Number of samples to generate."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=128, help="Number of steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--save_latent", action="store_true", help="Save latents.")
    parser.add_argument("--ddim", action="store_true", help="Save video.")
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
            scheduler_kwargs.pop("variance_type")
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

    # 5 - Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    generator = (
        torch.Generator(device=device).manual_seed(config.seed)
        if config.seed is not None
        else None
    )
    unet = unet.to(device, dtype)
    vae = vae.to(device, torch.float32)
    unet.eval()
    vae.eval()
    if text_encoder is not None:
        text_encoder = text_encoder.to(device)
        tokenizer = tokenizer.to(device)
        text_encoder.eval()

    format_input = padf
    format_output = unpadf

    B, C, H, W = (
        args.batch_size,
        config.unet.out_channels,
        config.unet.sample_size,
        config.unet.sample_size,
    )

    forward_kwargs = {
        "timestep": -1,
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
        conditioning_value = None  # For text, we'll handle differently

    sample_index = 0
    filelist = []

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    if args.save_latent:
        os.makedirs(os.path.join(args.output, "latents"), exist_ok=True)
    finished = False

    # 6 - Generate samples
    with torch.no_grad():
        for _ in tqdm(range(int(np.ceil(args.num_samples / args.batch_size)))):
            if finished:
                break

            latents = torch.randn(
                (B, C, H, W), device=device, dtype=dtype, generator=generator
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
            # Set the correct keyword argument based on conditioning type
            forward_kwargs["encoder_hidden_states"] = conditioning
            use_condition_guidance = args.condition_guidance_scale > 1.0
            with torch.autocast("cuda"):
                for t in timesteps:
                    forward_kwargs["timestep"] = t
                    latent_model_input = latents
                    latent_model_input = scheduler.scale_model_input(
                        latent_model_input, timestep=t
                    )
                    latent_model_input, padding = format_input(
                        latent_model_input, mult=3
                    )
                    if use_condition_guidance and args.sampling_mode == "diffusion":
                        # Classifier-free guidance
                        # Conditional prediction
                        forward_kwargs["encoder_hidden_states"] = conditioning
                        noise_pred_cond = unet(
                            latent_model_input, **forward_kwargs
                        ).sample
                        noise_pred_cond = format_output(noise_pred_cond, pad=padding)

                        # Unconditional prediction
                        forward_kwargs["encoder_hidden_states"] = torch.zeros_like(
                            conditioning
                        )
                        noise_pred_uncond = unet(
                            latent_model_input, **forward_kwargs
                        ).sample
                        noise_pred_uncond = format_output(
                            noise_pred_uncond, pad=padding
                        )

                        noise_pred = (
                            noise_pred_uncond
                            + args.condition_guidance_scale
                            * (noise_pred_cond - noise_pred_uncond)
                        )
                    else:
                        # No classifier-free guidance
                        noise_pred = unet(latent_model_input, **forward_kwargs).sample
                        noise_pred = format_output(noise_pred, pad=padding)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

            if args.save_latent:
                latents_clean = latents.clone()

            # VAE decode
            rep = [1, 3, 1, 1]
            latents = latents / vae.config.scaling_factor
            images = vae.decode(latents.float()).sample
            images = (images + 1) * 128  # [-1, 1] -> [0, 256]

            # grayscale
            images = images.mean(1).unsqueeze(1).repeat(*rep)

            images = images.clamp(0, 255).to(torch.uint8).cpu()
            images = rearrange(images, "b c h w -> b h w c")
            # Get conditioning values for metadata
            if args.conditioning_type == "class_id":
                cond_values = conditioning.squeeze().to(torch.int).tolist()
            elif args.conditioning_type == "view":
                cond_values = conditioning.squeeze().to(torch.int).tolist()
            else:  # text
                cond_values = text_conditioning

            # 7 - Save samples
            images = images.numpy()
            for j in range(B):
                # Save metadata
                filelist.append(
                    [
                        f"sample_{sample_index:06d}",
                        args.conditioning_type,
                        cond_values[j],
                    ]
                )
                Image.fromarray(images[j]).save(
                    os.path.join(
                        args.output, "images", f"sample_{sample_index:06d}.jpg"
                    )
                )
                if args.save_latent:
                    torch.save(
                        latents_clean[j].clone(),
                        os.path.join(
                            args.output, "latents", f"sample_{sample_index:06d}.pt"
                        ),
                    )

                sample_index += 1
                if sample_index >= args.num_samples:
                    finished = True
                    break
    # Save metadata
    df = pd.DataFrame(
        filelist,
        columns=["FileName", "CondType", "CondValue"],
    )
    df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)

    # Save generation parameters
    params = {
        "sampling_mode": args.sampling_mode,
        "conditioning_type": args.conditioning_type,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "condition_guidance_scale": args.condition_guidance_scale,
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "generation_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"Finished generating {sample_index} samples.")

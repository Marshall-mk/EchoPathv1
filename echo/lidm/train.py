import argparse
import logging
import math
import os
import shutil
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from tqdm.auto import tqdm
from packaging import version
from skimage.metrics import structural_similarity
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from echo.common.datasets import instantiate_dataset
from echo.common import padf, unpadf, instantiate_from_config, FlowMatchingScheduler

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

"""
CUDA_VISIBLE_DEVICES='2,3,4,5' accelerate launch  
	--num_processes 4  
	--multi_gpu   
	--mixed_precision fp16 
	-m  echo.lidm.train  
	--config echo/lidm/configs/default.yaml 
	--training_mode diffusion 
	--conditioning_type text
 	--condition_guidance_scale 5.0 
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


def compute_validation_metrics(generated_images, reference_images):
    """Compute validation metrics between generated and reference images."""
    metrics = {}

    # Calculate SSIM
    ssim_values = []
    for i in range(len(generated_images)):
        ssim_val = calculate_ssim(generated_images[i], reference_images[i])
        ssim_values.append(ssim_val)
    metrics["ssim"] = np.mean(ssim_values)

    # Calculate PSNR
    psnr_values = []
    for i in range(len(generated_images)):
        psnr_val = calculate_psnr(generated_images[i], reference_images[i])
        psnr_values.append(psnr_val)
    metrics["psnr"] = np.mean(psnr_values)

    # You could add other metrics like FID if you have a pretrained model

    return metrics


def calculate_ssim(generated_images, reference_image):
    """Calculate SSIM between two images."""
    # Convert to numpy arrays
    generated_images = generated_images.numpy()
    reference_image = reference_image.numpy()

    # Calculate SSIM for each frame
    ssim_values = []
    for i in range(generated_images.shape[0]):
        ssim_val = structural_similarity(
            generated_images[i], reference_image[i], multichannel=True
        )
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)


def calculate_psnr(generated_image, reference_image):
    """Calculate PSNR between two images."""
    # Convert to numpy arrays
    generated_image = generated_image.numpy()
    reference_image = reference_image.numpy()

    # Calculate PSNR for each frame
    psnr_values = []
    for i in range(generated_image.shape[0]):
        mse = np.mean((generated_image[i] - reference_image[i]) ** 2)
        if mse == 0:
            psnr_values.append(float("inf"))
        else:
            psnr_val = 20 * np.log10(255.0 / np.sqrt(mse))
            psnr_values.append(psnr_val)

    return np.mean(psnr_values)


def log_validation(
    config,
    unet,
    vae,
    scheduler,
    accelerator,
    weight_dtype,
    epoch,
    val_dataset,
    conditioning_type="text",
    text_encoder=None,
    tokenizer=None,
):
    logger.info("Running validation... ")

    val_unet = accelerator.unwrap_model(unet)
    val_vae = vae.to(accelerator.device, dtype=torch.float32)
    scheduler.set_timesteps(config.validation_timesteps)
    timesteps = scheduler.timesteps

    if config.enable_xformers_memory_efficient_attention:
        val_unet.enable_xformers_memory_efficient_attention()

    if config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    indices = np.random.choice(
        len(val_dataset), size=config.validation_count, replace=False
    )
    ref_elements = [val_dataset[i] for i in indices]
    ref_frames = [e["image"] for e in ref_elements]
    ref_frames = torch.stack(ref_frames, dim=0)  # B x C x H x W
    ref_frames = ref_frames.to(accelerator.device, dtype=weight_dtype)

    format_input = padf
    format_output = unpadf
    # Get conditioning based on type
    if conditioning_type == "class_id":
        conditioning = torch.tensor(
            [e["class_id"] for e in ref_elements],
            device=accelerator.device,
            dtype=weight_dtype,
        )
    elif conditioning_type == "lvef":
        conditioning = torch.tensor(
            [e["lvef"] for e in ref_elements],
            device=accelerator.device,
            dtype=weight_dtype,
        )
    elif conditioning_type == "view":
        conditioning = torch.tensor(
            [e["view"] for e in ref_elements],
            device=accelerator.device,
            dtype=weight_dtype,
        )
    elif conditioning_type == "text":
        # tokenize text inputs
        input_ids, attention_mask = tokenize_text(
            [e["text"] for e in ref_elements], tokenizer
        )
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)
        conditioning = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.to(dtype=weight_dtype)
    else:
        raise ValueError(f"Unsupported conditioning type: {conditioning_type}")

    # Reshape conditioning for model input
    if (
        conditioning_type != "text"
    ):  # text embeddings would already be in the right shape
        conditioning = conditioning[:, None, None]  # B -> B x 1 x 1

    logger.info("Sampling... ")
    with torch.no_grad(), torch.autocast("cuda"):
        # prepare model inputs
        B, C, H, W = (
            config.validation_count,
            4,
            config.unet.sample_size,
            config.unet.sample_size,
        )
        latents = torch.randn(
            (B, C, H, W),
            device=accelerator.device,
            dtype=weight_dtype,
            generator=generator,
        )
        condition_guidance_scale = getattr(config, "condition_guidance_scale", 5.0)
        # Set up for class guidance
        if condition_guidance_scale > 1.0:
            # For class guidance, create a no-class condition (either zeros or empty)
            if conditioning_type == "text":
                # For text, use empty text embeddings from the tokenizer
                empty_text = [""] * len(ref_elements)
                empty_ids, empty_mask = tokenize_text(empty_text, tokenizer)
                empty_ids = empty_ids.to(accelerator.device)
                empty_mask = empty_mask.to(accelerator.device)
                uncond_class = text_encoder(
                    input_ids=empty_ids, attention_mask=empty_mask
                ).last_hidden_state.to(dtype=weight_dtype)
            else:
                # For other condition types, use zeros
                uncond_class = torch.zeros_like(conditioning)

            # Concatenate class conditions with unconditional ones
            combined_class = torch.cat([uncond_class, conditioning])
        else:
            combined_class = conditioning

        # Sampling loop
        if config.training_mode == "diffusion":
            # reverse diffusionn loop
            for t in timesteps:
                latent_model_input = latents
                if condition_guidance_scale > 1.0:
                    latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )
                latent_model_input, padding = format_input(latent_model_input, mult=3)
                forward_kwargs = {
                    "timestep": t,
                    "encoder_hidden_states": combined_class,
                }
                noise_pred = unet(latent_model_input, **forward_kwargs).sample
                noise_pred = format_output(noise_pred, pad=padding)
                # Apply guidance
                if condition_guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    # Apply class guidance
                    noise_pred = noise_pred_uncond + condition_guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        else:
            # Flow matching sampling (simplified)
            t_steps = torch.linspace(
                1, 0, config.validation_timesteps, device=accelerator.device
            )
            dt = 1.0 / (config.validation_timesteps - 1)

            for i in range(len(t_steps) - 1):
                t = t_steps[i]
                t_tensor = t.repeat(B)

                # Prepare batched latents based on guidance configuration
                latent_model_input = latents

                if condition_guidance_scale > 1.0:
                    latent_model_input = torch.cat([latent_model_input] * 2)

                latent_model_input, padding = format_input(latent_model_input, mult=3)

                # Forward pass
                forward_kwargs = {
                    "timestep": t_tensor
                    if condition_guidance_scale <= 1.0
                    else t_tensor.repeat(2),
                    "encoder_hidden_states": combined_class,
                }

                velocity_pred = unet(latent_model_input, **forward_kwargs).sample
                velocity_pred = format_output(velocity_pred, pad=padding)

                # Apply guidance
                if condition_guidance_scale > 1.0:
                    velocity_pred_uncond, velocity_pred_cond = velocity_pred.chunk(2)
                    # Apply class guidance
                    velocity_pred = velocity_pred_uncond + condition_guidance_scale * (
                        velocity_pred_cond - velocity_pred_uncond
                    )

                # Euler step
                latents = latents - velocity_pred * dt

    # VAE decoding
    with torch.no_grad():  # no autocast
        latents = latents / val_vae.config.scaling_factor
        images = val_vae.decode(latents.float()).sample
        images = (images + 1) * 128  # [-1, 1] -> [0, 256]
        images = images.clamp(0, 255).to(torch.uint8).cpu()

        ref_frames = ref_frames / val_vae.config.scaling_factor
        ref_frames = val_vae.decode(ref_frames.float()).sample
        ref_frames = (ref_frames + 1) * 128  # [-1, 1] -> [0, 256]
        ref_frames = ref_frames.clamp(0, 255).to(torch.uint8).cpu()
        try:
            metrics = compute_validation_metrics(images, ref_frames)
        except Exception as e:
            logger.error(f"Error computing validation metrics: {e}")
            metrics = {}
        images = torch.cat(
            [ref_frames, images], dim=2
        )  # B x C x (2 H) x W // vertical concat

    # reshape for wandb
    images = rearrange(images, "b c h w -> h (b w) c")  # prepare for wandb
    images = images.numpy()

    logger.info("Done sampling... ")

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"validation": wandb.Image(images), **metrics})
            logger.info("Samples sent to wandb.")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del val_unet
    del val_vae
    torch.cuda.empty_cache()

    return images


def train(
    config,
    training_mode="diffusion",  # or "flow_matching"
    conditioning_type="text",  # or "lvef", "view", "text"
):
    # Setup accelerator
    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # Set up scheduler based on training mode
    if training_mode == "diffusion":
        noise_scheduler_kwargs = OmegaConf.to_container(
            config.noise_scheduler, resolve=True
        )
        noise_scheduler_klass_name = noise_scheduler_kwargs.pop("_class_name")
        noise_scheduler_klass = globals().get(noise_scheduler_klass_name, None)
        assert noise_scheduler_klass is not None, (
            f"Could not find class {noise_scheduler_klass_name}"
        )
        scheduler = noise_scheduler_klass(**noise_scheduler_kwargs)
    else:  # flow_matching
        scheduler = FlowMatchingScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 1000)
        )
    # Save training mode to config
    config.training_mode = training_mode

    # Load VAE
    vae = AutoencoderKL.from_pretrained(config.vae_path).cpu()

    # Create the video unet
    unet, unet_klass, unet_kwargs = instantiate_from_config(
        config.unet, ["diffusers"], return_klass_kwargs=True
    )
    noise_scheduler = instantiate_from_config(config.noise_scheduler, ["diffusers"])

    format_input = padf
    format_output = unpadf

    # setup text encoder and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path)
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    unet.train()

    if not config.train_text_encoder:  # freeze text encoder if not training it
        text_encoder.requires_grad_(False)
    else:
        text_encoder.train()

    # Create EMA for the unet.
    if config.use_ema:
        ema_unet = unet_klass(**unet_kwargs)
        ema_unet = EMAModel(
            ema_unet.parameters(), model_cls=unet_klass, model_config=ema_unet.config
        )
        # Create EMA for text encoder if it's being trained
        if config.train_text_encoder:
            ema_text_encoder = CLIPTextModel.from_pretrained(
                config.pretrained_model_name_or_path
            )
            ema_text_encoder = EMAModel(
                ema_text_encoder.parameters(),
                model_cls=CLIPTextModel,
                model_config=ema_text_encoder.config,
            )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    # Save UNet
                    if i == 0:  # First model is always UNet
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    # Save text encoder if it's being trained
                    elif (
                        config.train_text_encoder and i == 1
                    ):  # Second model is text encoder if it exists
                        model.save_pretrained(os.path.join(output_dir, "text_encoder"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), unet_klass
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()
                # Load text encoder if it exists and is being trained
                if (
                    i == 0
                    and config.train_text_encoder
                    and os.path.isdir(os.path.join(input_dir, "text_encoder"))
                ):
                    load_model = CLIPTextModel.from_pretrained(
                        input_dir, subfolder="text_encoder"
                    )
                    model.load_state_dict(load_model.state_dict())
                # Load UNet
                else:
                    load_model = unet_klass.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataset = instantiate_dataset(config.datasets, split=["TRAIN"])
    val_dataset = instantiate_dataset(config.datasets, split=["VAL"])

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # Prepare with accelerator
    if config.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # Move and prepare EMA models after accelerator preparation
    if config.use_ema:
        ema_unet.to(accelerator.device)
        if config.train_text_encoder:
            # Move text encoder EMA model to the same device as the text encoder
            ema_text_encoder.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(config, resolve=True)
        accelerator.init_trackers(
            config.tracker_project_name,
            tracker_config,
            init_kwargs={
                "wandb": {"group": config.wandb_group},
            },
        )

    # Train!
    total_batch_size = (
        config.train_batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    model_num_params = sum(p.numel() for p in unet.parameters())
    model_trainable_params = sum(
        p.numel() for p in unet.parameters() if p.requires_grad
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Training mode = {training_mode}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info(
        f"  U-Net: Total params = {model_num_params} \t Trainable params = {model_trainable_params} ({model_trainable_params / model_num_params * 100:.2f}%)"
    )
    # Set default values for guidance scales if not provided
    if not hasattr(config, "condition_guidance_scale"):
        config.condition_guidance_scale = 5.0  # default strong class guidance
    logger.info(f"Condition guidance scale: {config.condition_guidance_scale}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        # disable=not accelerator.is_local_main_process,
        disable=not accelerator.is_main_process,
    )

    for epoch in range(first_epoch, config.num_train_epochs):
        train_loss = 0.0
        prediction_mean = 0.0
        prediction_std = 0.0
        target_mean = 0.0
        target_std = 0.0
        mean_losses = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = batch["image"]  # B x C x H x W
                # Get conditioning based on type and convert to proper dtype
                if conditioning_type == "class_id":
                    conditioning = batch["class_id"].to(dtype=weight_dtype)
                elif conditioning_type == "lvef":
                    conditioning = batch["lvef"].to(dtype=weight_dtype)
                elif conditioning_type == "view":
                    conditioning = batch["view"].to(dtype=weight_dtype)
                elif conditioning_type == "text":
                    # tokenize text inputs
                    input_ids, attention_mask = tokenize_text(batch["text"], tokenizer)
                    # Move tensors to the correct device
                    input_ids = input_ids.to(accelerator.device)
                    attention_mask = attention_mask.to(accelerator.device)

                    # encode text inputs through CLIP
                    # The correct way to extract hidden states from CLIP text encoder
                    text_outputs = text_encoder(
                        input_ids, attention_mask=attention_mask
                    )
                    # Properly extract the hidden states and convert to proper dtype
                    conditioning = text_outputs.last_hidden_state.to(dtype=weight_dtype)
                else:
                    raise ValueError(
                        f"Unsupported conditioning type: {conditioning_type}"
                    )

                B, C, H, W = latents.shape
                if (
                    conditioning_type != "text"
                ):  # text embeddings would already be in the right shape
                    conditioning = conditioning[:, None, None]

                # Class conditioning dropout (for class-free guidance)
                class_conditioning_mask = torch.rand_like(
                    conditioning[:, 0:1, 0:1],
                    device=accelerator.device,
                    dtype=weight_dtype,
                ) > config.get("drop_conditioning", 0.3)
                conditioning = conditioning * class_conditioning_mask

                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    int(noise_scheduler.config.num_train_timesteps),
                    (B,),
                    device=latents.device,
                ).long()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.noise_offset > 0:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )
                # Add noise to inputs
                if training_mode == "diffusion":
                    if config.get("input_perturbation", 0) > 0.0:
                        noisy_latents = noise_scheduler.add_noise(
                            latents,
                            noise
                            + config.input_perturbation
                            * torch.rand(
                                1,
                            ).item()
                            * torch.randn_like(noise),
                            timesteps,
                        )
                    else:
                        noisy_latents = noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )
                else:
                    # Flow matching process
                    t = timesteps.float() / scheduler.config.num_train_timesteps
                    t = t.view(-1, 1, 1, 1, 1)
                    noisy_latents = (1 - t) * latents + t * noise

                # Predict the noise residual and compute loss
                noisy_latents, padding = format_input(noisy_latents, mult=3)
                # Forward pass
                forward_kwargs = {
                    "timestep": timesteps,
                    "encoder_hidden_states": conditioning,
                }
                model_pred = unet(sample=noisy_latents, **forward_kwargs).sample
                model_pred = format_output(model_pred, pad=padding)
                # Set target based on training mode
                if training_mode == "diffusion":
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        assert noise_scheduler.config.prediction_type == "sample", (
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )
                        target = latents
                else:  # flow_matching
                    # Target is the normalized direction from noise to clean sample
                    target = latents - noise

                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean()
                mean_loss = loss.item()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(config.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                mean_losses += mean_loss / config.gradient_accumulation_steps
                prediction_mean += (
                    model_pred.mean().item() / config.gradient_accumulation_steps
                )
                prediction_std += (
                    model_pred.std().item() / config.gradient_accumulation_steps
                )
                target_mean += target.mean().item() / config.gradient_accumulation_steps
                target_std += target.std().item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.use_ema:
                    ema_unet.step(unet.parameters())
                    if config.train_text_encoder:
                        # Ensure parameters are on the same device before EMA update
                        ema_text_encoder.to(accelerator.device)
                        # Get unwrapped text encoder if using accelerator
                        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                        ema_text_encoder.step(unwrapped_text_encoder.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "prediction_mean": prediction_mean,
                        "prediction_std": prediction_std,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "mean_losses": mean_losses,
                    },
                    step=global_step,
                )
                train_loss = 0.0
                prediction_mean = 0.0
                prediction_std = 0.0
                target_mean = 0.0
                target_std = 0.0
                mean_losses = 0.0

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - config.checkpoints_total_limit
                                    + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        config.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            config.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        OmegaConf.save(
                            config, os.path.join(config.output_dir, "config.yaml")
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % config.validation_steps == 0:
                    if config.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                        if config.train_text_encoder:
                            # Ensure text encoder EMA is on the correct device
                            ema_text_encoder.to(accelerator.device)
                            # Get unwrapped text encoder
                            unwrapped_text_encoder = accelerator.unwrap_model(
                                text_encoder
                            )
                            ema_text_encoder.store(unwrapped_text_encoder.parameters())
                            ema_text_encoder.copy_to(
                                unwrapped_text_encoder.parameters()
                            )

                    log_validation(
                        config,
                        unet,
                        vae,
                        deepcopy(noise_scheduler),
                        accelerator,
                        weight_dtype,
                        epoch,
                        val_dataset,
                        conditioning_type=conditioning_type,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                    )

                    if config.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
                        if config.train_text_encoder:
                            # Get unwrapped text encoder
                            unwrapped_text_encoder = accelerator.unwrap_model(
                                text_encoder
                            )
                            ema_text_encoder.restore(
                                unwrapped_text_encoder.parameters()
                            )
    # Save the final model
    if accelerator.is_main_process:
        # Create the pipeline using the trained modules
        unet = accelerator.unwrap_model(unet)
        if config.use_ema:
            ema_unet.copy_to(unet.parameters())

        # Save text encoder if it was trained
        if config.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
            if (
                config.use_ema
                and hasattr(config, "ema_text_encoder")
                and config.ema_text_encoder
            ):
                ema_text_encoder.copy_to(text_encoder.parameters())
            text_encoder.save_pretrained(
                os.path.join(config.output_dir, "text_encoder")
            )
            tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))

        # Always save the UNet
        unet.save_pretrained(os.path.join(config.output_dir, "unet"))

        # Save the config
        OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
        logger.info(f"Saved models and config to {config.output_dir}")

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion or flow matching model with different conditioning options"
    )
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="diffusion",
        choices=["diffusion", "flow_matching"],
        help="Training methodology to use",
    )
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="class_id",
        choices=["class_id", "lvef", "view", "text"],
        help="Type of conditioning to use",
    )
    parser.add_argument(
        "--condition_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for class conditioning (1.0=no guidance, recommended range: 1.0-10.0, higher means stronger influence)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Set guidance scale parameters from command line args
    config.condition_guidance_scale = args.condition_guidance_scale
    # Set default validation samples if not in config
    if not hasattr(config, "num_validation_samples"):
        config.num_validation_samples = 4

    # Set default paths for text encoder and tokenizer if not in config
    if args.conditioning_type == "text" and (
        not hasattr(config, "text_encoder_path")
        or not hasattr(config, "tokenizer_path")
    ):
        config.text_encoder_path = "openai/clip-vit-large-patch14"
        config.pretrained_model_name_or_path = "openai/clip-vit-large-patch14"
        config.tokenizer_path = "openai/clip-vit-large-patch14"
        config.train_text_encoder = True

    train(
        config,
        training_mode=args.training_mode,
        conditioning_type=args.conditioning_type,
    )

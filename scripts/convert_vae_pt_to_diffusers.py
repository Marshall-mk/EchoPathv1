import argparse
import io

import requests
import torch
from omegaconf import OmegaConf
import yaml

from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)


def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict[
        "encoder.conv_out.weight"
    ]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict[
        "encoder.norm_out.weight"
    ]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict[
        "encoder.norm_out.bias"
    ]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict[
        "decoder.conv_out.weight"
    ]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict[
        "decoder.norm_out.weight"
    ]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict[
        "decoder.norm_out.bias"
    ]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "encoder.down" in layer
        }
    )
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "decoder.up" in layer
        }
    )
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = (
                vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = (
                vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = (
                vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            )
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = (
                vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def infer_vae_config_from_checkpoint(checkpoint):
    """Infer VAE configuration from checkpoint state dict."""
    # Get channel dimensions from the checkpoint
    conv_in_channels = checkpoint.get("encoder.conv_in.weight", checkpoint.get("first_stage_model.encoder.conv_in.weight")).shape[1]
    
    # Based on your config: embed_dim: 4, double_z: True
    # This means latent_channels = embed_dim = 4, but encoder outputs 2*embed_dim = 8
    embed_dim = 4  # from your config
    latent_channels = embed_dim  # this is the actual latent dimension
    
    # The encoder conv_out produces 2*embed_dim channels due to double_z
    # The quant_conv maps from 2*embed_dim to 2*embed_dim  
    # The post_quant_conv maps from embed_dim to embed_dim
    
    # Based on your config: ch: 128, ch_mult: [1,2,2,4]
    base_channels = 128
    ch_mult = [1, 2, 4] # [1, 2, 2, 4]
    # Count down blocks
    down_block_indices = set()
    for key in checkpoint.keys():
        if "encoder.down." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "down" and i + 1 < len(parts):
                    try:
                        down_block_indices.add(int(parts[i + 1]))
                    except ValueError:
                        continue
    
    num_down_blocks = len(down_block_indices) if down_block_indices else len(ch_mult)
    
    # Create config matching your training config
    config = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.27.2",
        "act_fn": "silu",
        "block_out_channels": [base_channels * mult for mult in ch_mult],
        "down_block_types": ["DownEncoderBlock2D"] * num_down_blocks,
        "in_channels": conv_in_channels,
        "latent_channels": latent_channels,  # 4 from embed_dim
        "layers_per_block": 2,  # matches num_res_blocks: 2
        "norm_num_groups": 32,
        "out_channels": conv_in_channels,
        "sample_size": 112,  # matches resolution: 112 from your config
        "scaling_factor": 0.18215,
        "up_block_types": ["UpDecoderBlock2D"] * num_down_blocks,
    }
    
    return config


def vae_pt_to_vae_diffuser(
    checkpoint_path: str,
    output_path: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.endswith("safetensors"):
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"]

    # Convert the VAE model.
    vae_config = infer_vae_config_from_checkpoint(checkpoint)
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vae_pt_path",
        default=None,
        type=str,
        required=True,
        help="Path to the VAE.pt to convert.",
    )
    parser.add_argument(
        "--dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the VAE.pt to convert.",
    )

    args = parser.parse_args()

    vae_pt_to_vae_diffuser(args.vae_pt_path, args.dump_path)

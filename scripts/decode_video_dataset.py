import argparse
import os
from pathlib import Path
from glob import glob
from einops import rearrange
import numpy as np
from tqdm.auto import tqdm
import cv2
import pandas as pd

import torch

from echosyn.common import load_model

"""
usage example:

python scripts/decode_video_dataset.py \
    -m models/vae \
    -i data/latents/dynamic \
    -o datasets/decoded_videos \
    -f
"""


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.latent_files = sorted(
            glob(os.path.join(folder, "**/*.pt"), recursive=True)
        )

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = self.latent_files[idx]
        latents = torch.load(latent_path)
        return latents, latent_path


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to model folder"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input folder containing latent files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output folder for decoded videos",
    )
    parser.add_argument(
        "-g",
        "--gray_scale",
        action="store_true",
        help="Convert output to gray scale",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing videos",
        default=False,
    )
    args = parser.parse_args()

    # Prepare
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    latents_in_folder = os.path.abspath(os.path.join(args.input, "Latents"))
    video_out_folder = os.path.abspath(os.path.join(args.output, "Videos"))

    # Check if FileList.csv exists to get FPS values
    fps_dict = {}
    filelist_path = os.path.join(args.input, "FileList.csv")
    if os.path.exists(filelist_path):
        df = pd.read_csv(filelist_path)
        if "FPS" in df.columns and "FileName" in df.columns:
            for _, row in df.iterrows():
                fps_dict[row["FileName"]] = row["FPS"]
            print(f"Loaded FPS information for {len(fps_dict)} files")

    os.makedirs(video_out_folder, exist_ok=True)

    print("Loading latents from ", latents_in_folder)
    print("Saving videos to ", video_out_folder)

    # Load VAE
    vae = load_model(args.model)
    vae = vae.to(device)
    vae.eval()

    # Load Dataset
    ds = LatentDataset(latents_in_folder)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)
    print(f"Found {len(ds)} latent files")

    batch_size = 32  # number of frames to decode simultaneously

    for latents, latent_path in tqdm(dl):
        latents = latents[0]  # Remove batch dimension
        latent_path = latent_path[0]  # Get string from list

        # output path
        rel_path = os.path.relpath(latent_path, latents_in_folder)
        opath = os.path.join(video_out_folder, rel_path.replace(".pt", ".avi"))

        # Get filename to lookup FPS
        fname = os.path.basename(latent_path).split(".")[0]
        fps = fps_dict.get(fname, 30)  # Default to 30 if not found

        # check if already exists
        if os.path.exists(opath) and not args.force_overwrite:
            print(f"Skipping {latent_path} as {opath} already exists")
            continue

        # decode latents
        all_frames = []
        for i in range(0, len(latents), batch_size):
            batch = latents[i : i + batch_size].to(device)
            with torch.no_grad():
                # Scale latents back before decoding
                scaled_latents = batch / vae.config.scaling_factor
                decoded = vae.decode(scaled_latents).sample

            # Convert to pixel values [0, 255]
            decoded = ((decoded + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            decoded = rearrange(decoded, "b c h w -> b h w c")  # B C H W -> B H W C

            # Convert to grayscale if needed
            if args.gray_scale:
                # Create grayscale version but keep 3 channels for video compatibility
                decoded_gray = decoded.mean(dim=3, keepdim=True).repeat(1, 1, 1, 3)
                decoded = decoded_gray

            decoded = decoded.cpu().numpy()
            all_frames.append(decoded)

        video = np.concatenate(all_frames, axis=0)

        # save as video
        os.makedirs(os.path.dirname(opath), exist_ok=True)

        # Get video dimensions
        height, width = video.shape[1], video.shape[2]

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(opath, fourcc, fps, (width, height))

        # Write frames
        for frame in video:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    print("Done")

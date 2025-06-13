import os, sys
import numpy as np
import pandas as pd
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision import transforms as T
from torchvision.transforms.functional import center_crop

from PIL import Image
import decord

decord.bridge.set_bridge("torch")


# support video and image + additional info (lvef, view, etc)
class EchoDynamic(Dataset):
    def __init__(
        self, config, split=["TRAIN", "VAL", "TEST"], datafolder="Videos", ext=".avi"
    ) -> None:
        super().__init__()
        # the config here is only the config for this dataset, ie config.dataset.dynamic

        if type(split) == str:
            split = [split]
        assert [s in ["TRAIN", "VAL", "TEST"] for s in split], (
            "Splits must be a list of TRAIN, VAL, TEST"
        )

        # assert type(config.target_fps) == int or config.target_fps in ["original", "random", "exponential"], "target_fps must be an integer, 'original', 'random' or 'exponential'"
        self.target_fps = config.target_fps
        # self.duration_seconds = config.target_duration
        self.resolution = config.target_resolution
        self.outputs = config.outputs
        if type(self.outputs) == str:
            self.outputs = [self.outputs]
        assert [o in ["video", "image", "lvef", "text"] for o in self.outputs], (
            "Outputs must be a list of video, image, lvef, text"
        )

        # Add text template configuration
        self.text_template = config.get(
            "text_template",
            "An echocardiography video with Left Ventricular Ejection Fraction {}%",
        )

        # self.duration_frames = int(self.target_fps * self.duration_seconds)
        self.duration_frames = config.target_nframes
        self.duration_seconds = (
            self.duration_frames / self.target_fps
            if type(self.target_fps) == int
            else None
        )

        # LOAD DATA
        assert hasattr(config, "root"), "No root folder specified in config"
        assert os.path.exists(os.path.join(config.root, datafolder)), (
            f"Data folder {os.path.join(config.root, datafolder)} does not exist"
        )
        assert os.path.exists(os.path.join(config.root, "FileList.csv")), (
            f"FileList.csv does not exist in {config.root}"
        )
        self.metadata = pd.read_csv(os.path.join(config.root, "FileList.csv"))
        self.metadata = self.metadata[
            self.metadata["Split"].isin(split)
        ]  # filter by split
        self.len_before_filter = len(self.metadata)
        # add duration column
        self.metadata["Duration"] = (
            self.metadata["NumberOfFrames"] / self.metadata["FPS"]
        )  # won't work for pediatrics
        # filter by duration
        if self.duration_seconds is not None:
            self.metadata = self.metadata[
                self.metadata["Duration"] > self.duration_seconds
            ]

        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: os.path.join(config.root, datafolder, x)
            if x.endswith(ext)
            else os.path.join(config.root, datafolder, x.split(".")[0] + ext)
        )
        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: os.path.exists(x)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]
        self.metadata.reset_index(inplace=True, drop=True)
        if len(self.metadata) == 0:
            raise ValueError(
                f"No data found in folder {os.path.join(config.root, datafolder)}"
            )

        self.transform = lambda x: x
        if hasattr(config, "transforms"):
            transforms = []
            for transform in config.transforms:
                tklass = getattr(T, transform.name)
                tobj = tklass(**transform.params)
                transforms.append(tobj)
            self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx, return_row=False):
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
            "still": False,
        }

        if "image" in self.outputs or "video" in self.outputs or "key_frames" in self.outputs:
            reader = decord.VideoReader(
                row["VideoPath"],
                ctx=decord.cpu(),
                width=self.resolution,
                height=self.resolution,
            )
            og_fps = reader.get_avg_fps()
            og_frame_count = len(reader)

        if "video" in self.outputs:
            # Generate indices to resample
            # Generate a random starting point to cover all frames
            if self.target_fps == "original":
                target_fps = og_fps
            elif self.target_fps == "random":
                target_fps = np.random.randint(16, 120)
            elif self.target_fps == "half":
                target_fps = int(og_fps // 2)
            elif self.target_fps == "exponential":
                rnd, offset = np.random.randint(0, 100), 11
                target_fps = int(np.exp(rnd / offset) + offset)  # min: 12, max: ~8000
            else:
                target_fps = self.target_fps
            new_frame_count = np.floor(target_fps / og_fps * og_frame_count).astype(int)
            resample_indices_a = (
                np.linspace(0, og_frame_count - 1, new_frame_count, endpoint=False)
                .round()
                .astype(int)
            )
            start_idx = (
                np.random.choice(np.arange(0, resample_indices_a[1]))
                if len(resample_indices_a) > 1 and resample_indices_a[1] > 1
                else 0
            )
            resample_indices_a = resample_indices_a + start_idx

            # Sample a random chunk to cover the requested duration
            start_idx = (
                np.random.choice(
                    np.arange(0, len(resample_indices_a) - self.duration_frames)
                )
                if len(resample_indices_a) > self.duration_frames
                else 0
            )
            end_idx = start_idx + self.duration_frames
            resample_indices_b = resample_indices_a[start_idx:end_idx]
            resample_indices_b = resample_indices_b[
                resample_indices_b < len(reader)
            ]  # remove indices that are out of bounds
            video = reader.get_batch(resample_indices_b)  # T x H x W x C, uint8

            # Check if padding is needed
            p_index = len(video)
            if len(video) < self.duration_frames:
                padding_element = torch.zeros_like(video[0])
                padding = torch.stack(
                    [padding_element] * (self.duration_frames - len(video))
                )
                video = torch.cat((video, padding), dim=0)
                assert len(video) == self.duration_frames, (
                    f"Video length is {len(video)} but should be {self.duration_frames}"
                )

            video = video.float() / 128.0 - 1  # normalize to [-1, 1]
            video = video.permute(3, 0, 1, 2)  # T x H x W x C -> C x T x H x W
            output["video"] = self.transform(video)
            output["fps"] = target_fps
            output["padding"] = p_index
            if self.target_fps == "exponential":
                resample_indices_b[1:] = (
                    resample_indices_b[1:] - resample_indices_b[:-1] >= 1
                ).cumsum(0)
                resample_indices_b[0] = 0
                output["indices"] = np.concatenate(
                    (
                        resample_indices_b,
                        np.repeat(
                            resample_indices_b[-1],
                            self.duration_frames - len(resample_indices_b),
                        ),
                    )
                )

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0  # normalize to [0, 1]
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        if "text" in self.outputs and "EF" in row:
            output["text"] = self.text_template.format(int(row["EF"]))

        if "image" in self.outputs:
            image = reader.get_batch(np.random.randint(0, og_frame_count, 1))[
                0
            ]  # H x W x C, uint8
            image = image.float() / 128.0 - 1
            image = image.permute(2, 0, 1)  # H x W x C -> C x H x W
            output["image"] = self.transform(image)

        if "key_frames" in self.outputs:
            key_frame_columns = ["Start_ED", "ES", "End_ED"]
            key_frames = []

            for col in key_frame_columns:
                if col in row and pd.notna(row[col]):
                    frame_idx = int(row[col])
                    # Ensure frame index is within bounds
                    if 0 <= frame_idx < og_frame_count:
                        frame = reader.get_batch([frame_idx])[0]  # H x W x C, uint8
                        frame = frame.float() / 128.0 - 1  # normalize to [-1, 1]
                        frame = frame.permute(2, 0, 1)  # H x W x C -> C x H x W
                        key_frames.append(frame)
                    else:
                        # Use first frame as fallback if index is out of bounds
                        frame = reader.get_batch([0])[0]
                        frame = frame.float() / 128.0 - 1
                        frame = frame.permute(2, 0, 1)
                        key_frames.append(frame)
                else:
                    # Use first frame as fallback if column is missing or NaN
                    frame = reader.get_batch([0])[0]
                    frame = frame.float() / 128.0 - 1
                    frame = frame.permute(2, 0, 1)
                    key_frames.append(frame)

            if key_frames:
                # Stack key frames channel-wise: (3 frames x C x H x W) -> (3*C x H x W)
                output["key_frames"] = torch.cat(key_frames, dim=0)
            else:
                # Fallback: use first frame repeated 3 times
                first_frame = reader.get_batch([0])[0]
                first_frame = first_frame.float() / 128.0 - 1
                first_frame = first_frame.permute(2, 0, 1)
                output["key_frames"] = torch.cat([first_frame] * 3, dim=0)

        if return_row:
            return output, row

        return output


class EchoDynamicLatent(EchoDynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        self.config = config

        super().__init__(config, split, datafolder="Latents", ext=".pt")

        self.view = config.get("views", "ALL")  # A4C, PSAX, ALL
        if self.view == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["View"] == self.view]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for view {self.view}")

    def __getitem__(self, idx, return_row=False):
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
        }

        if (
            "image" in self.outputs
            or "video" in self.outputs
            or "key_frames" in self.outputs
        ):
            latent_file = row["VideoPath"]
            latent_video_tensor = torch.load(latent_file)  # T x C x H x W
            og_fps = row["FPS"]
            og_frame_count = len(latent_video_tensor)

        if "video" in self.outputs:
            if self.target_fps == "original":
                target_fps = og_fps
            elif self.target_fps == "random":
                target_fps = np.random.randint(8, 50)
            else:
                target_fps = self.target_fps

            new_frame_count = np.floor(target_fps / og_fps * og_frame_count).astype(int)
            resample_indices = (
                np.linspace(0, og_frame_count, new_frame_count, endpoint=False)
                .round()
                .astype(int)
            )
            start_idx = (
                np.random.choice(np.arange(0, resample_indices[1]))
                if len(resample_indices) > 1 and resample_indices[1] > 1
                else 0
            )
            resample_indices = resample_indices + start_idx

            # Sample a random chunk to cover the requested duration
            start_idx = (
                np.random.choice(
                    np.arange(0, len(resample_indices) - self.duration_frames)
                )
                if len(resample_indices) > self.duration_frames
                else 0
            )
            end_idx = start_idx + self.duration_frames
            resample_indices = resample_indices[start_idx:end_idx]
            resample_indices = resample_indices[
                resample_indices < og_frame_count
            ]  # remove indices that are out of bounds

            latent_video_sample = latent_video_tensor[resample_indices]

            # Check if padding is needed
            p_index = len(latent_video_sample)
            if len(latent_video_sample) < self.duration_frames:
                padding_element = torch.zeros_like(latent_video_sample[0])
                padding = torch.stack(
                    [padding_element]
                    * (self.duration_frames - len(latent_video_sample))
                )
                latent_video_sample = torch.cat((latent_video_sample, padding), dim=0)
                assert len(latent_video_sample) == self.duration_frames, (
                    f"Video length is {len(latent_video_sample)} but should be {self.duration_frames}"
                )

            latent_video_sample = latent_video_sample.permute(
                1, 0, 2, 3
            )  # T x C x H x W -> C x T x H x W
            output["video"] = self.transform(latent_video_sample)
            output["fps"] = target_fps
            output["padding"] = p_index

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        if "text" in self.outputs and "EF" in row:
            output["text"] = self.text_template.format(int(row["EF"]))

        if "image" in self.outputs:
            latent_image_tensor = latent_video_tensor[
                np.random.randint(0, og_frame_count, 1)
            ][0]  # C x H x W
            output["image"] = self.transform(latent_image_tensor)

        if "key_frames" in self.outputs:
            key_frame_columns = ["Start_ED", "ES", "End_ED"]
            key_frames = []

            for col in key_frame_columns:
                if col in row and pd.notna(row[col]):
                    frame_idx = int(row[col])
                    # Ensure frame index is within bounds
                    if 0 <= frame_idx < og_frame_count:
                        key_frames.append(latent_video_tensor[frame_idx])  # C x H x W
                    else:
                        # Use first frame as fallback if index is out of bounds
                        key_frames.append(latent_video_tensor[0])
                else:
                    # Use first frame as fallback if column is missing or NaN
                    key_frames.append(latent_video_tensor[0])

            if key_frames:
                # Stack key frames channel-wise: (3 frames x C x H x W) -> (3*C x H x W)
                output["key_frames"] = torch.cat(key_frames, dim=0)
            else:
                # Fallback: use first frame repeated 3 times
                first_frame = latent_video_tensor[0]
                output["key_frames"] = torch.cat([first_frame] * 3, dim=0)

        if return_row:
            return output, row

        return output


class EchoDynamicLatentAR(Dataset):
    def __init__(
        self,
        config=None,
        datafolder="Latents",
        ext=".pt",
        target_resolution=14,
        target_fps=32,
        target_nframes=64,
        text_template="An echocardiography video with Left Ventricular Ejection Fraction {}%",
        outputs=["text", "prior_frame", "target_frame"],
        split=["TRAIN", "VAL", "TEST"],
        view="ALL",
        prior_sequence_length=64,  # Length of prior sequence
        target_sequence_length=64,  # Length of target sequence
        stride_between_sequences=32,  # Frames to skip between prior and target sequences
    ) -> None:
        # Initialize without calling _build_frame_pairs yet
        super().__init__()
        self.config = config
        self.target_fps = target_fps
        self.resolution = target_resolution
        self.outputs = outputs
        self.duration_frames = target_nframes
        self.duration_seconds = (
            self.duration_frames / self.target_fps
            if type(self.target_fps) == int
            else None
        )
        self.text_template = text_template
        self.root = config.root
        self.datafolder = datafolder
        self.ext = ext

        # 3D sequence parameters
        self.prior_sequence_length = prior_sequence_length
        self.target_sequence_length = target_sequence_length
        self.stride_between_sequences = stride_between_sequences

        self.transform = T.Compose(
            [T.Resize((self.resolution, self.resolution)), T.ToTensor()]
        )

        # LOAD DATA
        if type(split) == str:
            split = [split]
        assert [s in ["TRAIN", "VAL", "TEST"] for s in split], (
            "Splits must be a list of TRAIN, VAL, TEST"
        )
        if type(self.outputs) == str:
            self.outputs = [self.outputs]
        # print(f"Outputs: {self.root}")
        assert all(
            [o in ["text", "prior_frame", "target_frame"] for o in self.outputs]
        ), "Outputs must be a list of text, prior_frame, target_frame"
        assert os.path.exists(os.path.join(self.root, self.datafolder)), (
            f"Data folder {os.path.join(self.root, self.datafolder)} does not exist"
        )
        assert os.path.exists(os.path.join(self.root, "FileList.csv")), (
            f"FileList.csv does not exist in {self.root}"
        )

        self.metadata = pd.read_csv(os.path.join(self.root, "FileList.csv"))

        # filter by split
        self.metadata = self.metadata[self.metadata["Split"].isin(split)]
        self.len_before_filter = len(self.metadata)

        # add duration column
        self.metadata["Duration"] = (
            self.metadata["NumberOfFrames"] / self.metadata["FPS"]
        )  # won't work for pediatrics

        # filter by duration (we want videos with at least duration_frames)
        if self.duration_seconds is not None:
            self.metadata = self.metadata[
                self.metadata["Duration"] >= self.duration_seconds
            ]

        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: os.path.join(self.root, self.datafolder, x)
            if x.endswith(ext)
            else os.path.join(self.root, self.datafolder, x.split(".")[0] + ext)
        )
        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: os.path.exists(x)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]
        self.metadata.reset_index(inplace=True, drop=True)
        if len(self.metadata) == 0:
            raise ValueError(
                f"No data found in folder {os.path.join(self.root, self.datafolder)}"
            )

        # filter by view
        self.view = view  # A4C, PSAX, ALL
        if self.view == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["View"] == self.view]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for view {self.view}")

        # Build sequence pairs for latent videos
        self._build_sequence_pairs_latent()

    def _build_sequence_pairs_latent(self):
        """Build sequence pairs for 3D model training with latent videos"""
        self.sequence_pairs = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            latent_file = row["VideoPath"]
            lvef = row["EF"]
            lvef_text = self.text_template.format(int(lvef))

            try:
                latent_video_tensor = torch.load(latent_file)  # T x C x H x W
                num_frames = len(latent_video_tensor)
                # print(f"Loaded {latent_file} with shape {latent_video_tensor.shape}, frames: {num_frames}")

                # Need at least prior_sequence_length + stride + target_sequence_length frames
                min_frames_needed = (
                    self.prior_sequence_length + self.stride_between_sequences
                )  # + self.target_sequence_length

                if num_frames < min_frames_needed:
                    # print(
                    #     f"Skipping {latent_file}: not enough frames ({num_frames} < {min_frames_needed})"
                    # )
                    continue

                # Create sequence pairs
                # For each possible starting point of the prior sequence
                for start_idx in range(0, num_frames - min_frames_needed + 1):
                    prior_end_idx = start_idx + self.prior_sequence_length
                    target_start_idx = start_idx + self.stride_between_sequences
                    target_end_idx = target_start_idx + self.target_sequence_length

                    # Ensure we don't exceed the video length
                    if target_end_idx <= num_frames:
                        self.sequence_pairs.append(
                            {
                                "video_idx": idx,
                                "prior_start_idx": start_idx,
                                "prior_end_idx": prior_end_idx,
                                "target_start_idx": target_start_idx,
                                "target_end_idx": target_end_idx,
                                "lvef_text": lvef_text,
                                "lvef": lvef / 100.0,
                            }
                        )
            except Exception as e:
                print(f"Error loading {latent_file}: {e}")
                continue

        print(
            f"Created {len(self.sequence_pairs)} sequence pairs from {len(self.metadata)} videos"
        )

    def __len__(self):
        return len(self.sequence_pairs)

    def __getitem__(self, idx):
        pair = self.sequence_pairs[idx]
        video_idx = pair["video_idx"]
        prior_start_idx = pair["prior_start_idx"]
        prior_end_idx = pair["prior_end_idx"]
        target_start_idx = pair["target_start_idx"]
        target_end_idx = pair["target_end_idx"]
        lvef_text = pair["lvef_text"]
        lvef = pair["lvef"]

        row = self.metadata.iloc[video_idx]
        latent_video_tensor = torch.load(row["VideoPath"])  # T x C x H x W

        # Get the sequence pairs
        prior_sequence = latent_video_tensor[
            prior_start_idx:prior_end_idx
        ]  # prior_sequence_length x C x H x W
        target_sequence = latent_video_tensor[
            target_start_idx:target_end_idx
        ]  # target_sequence_length x C x H x W

        # Rearrange to C x T x H x W format expected by 3D models
        prior_sequence = prior_sequence.permute(
            1, 0, 2, 3
        )  # C x prior_sequence_length x H x W
        target_sequence = target_sequence.permute(
            1, 0, 2, 3
        )  # C x target_sequence_length x H x W

        output = {
            "filename": row["FileName"],
            "prior_frames": prior_sequence,  # Now a sequence C x T x H x W
            "target_frames": target_sequence,  # Now a sequence C x T x H x W
            "text": lvef_text,
            # "lvef": torch.tensor(lvef, dtype=torch.float32),
        }
        if "view" in self.outputs and "View" in row:
            output["view"] = row["View"]
            # Include view in text conditioning
            output["text"] = f"{output['text']}, View: {row['View']}"

        return output


class EchoPediatric(EchoDynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split)

        # View
        self.view = config.get("views", "ALL")  # A4C, PSAX, ALL
        if self.view == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["View"] == self.view]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for view {self.view}")

    def __getitem__(self, idx):
        output, row = super().__getitem__(idx, return_row=True)
        if "view" in output:
            output["view"] = row["View"]

        if "text" in output:
            # Include view in text if available
            ef_text = self.text_template.format(int(row["EF"]))
            if "View" in row:
                output["text"] = f"{ef_text}, View: {row['View']}"
            else:
                output["text"] = ef_text

        return output


class EchoPediatricLatent(EchoDynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        self.config = config

        # Default text template for pediatric condition
        self.text_template = config.get(
            "text_template",
            "An echocardiography video with Left Ventricular Ejection Fraction {}%",
        )

        super().__init__(config, split, datafolder="Latents", ext=".pt")

        # View
        self.view = config.get("views", "ALL")  # A4C, PSAX, ALL
        if self.view == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["View"] == self.view]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for view {self.view}")

    def __getitem__(self, idx, return_row=False):
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
        }

        if (
            "image" in self.outputs
            or "video" in self.outputs
            or "key_frames" in self.outputs
        ):
            latent_file = row["VideoPath"]
            latent_video_tensor = torch.load(latent_file)
            # T x C x H x W
            og_fps = row["FPS"]
            og_frame_count = len(latent_video_tensor)
        if "video" in self.outputs:
            if self.target_fps == "original":
                target_fps = og_fps
            elif self.target_fps == "random":
                target_fps = np.random.randint(8, 50)
            else:
                target_fps = self.target_fps

            new_frame_count = np.floor(target_fps / og_fps * og_frame_count).astype(int)
            resample_indices = (
                np.linspace(0, og_frame_count, new_frame_count, endpoint=False)
                .round()
                .astype(int)
            )
            start_idx = (
                np.random.choice(np.arange(0, resample_indices[1]))
                if len(resample_indices) > 1 and resample_indices[1] > 1
                else 0
            )
            resample_indices = resample_indices + start_idx

            # Sample a random chunk to cover the requested duration
            start_idx = (
                np.random.choice(
                    np.arange(0, len(resample_indices) - self.duration_frames)
                )
                if len(resample_indices) > self.duration_frames
                else 0
            )
            end_idx = start_idx + self.duration_frames
            resample_indices = resample_indices[start_idx:end_idx]
            resample_indices = resample_indices[
                resample_indices < og_frame_count
            ]  # remove indices that are out of bounds
            latent_video_sample = latent_video_tensor[resample_indices]
            # Check if padding is needed
            p_index = len(latent_video_sample)
            if len(latent_video_sample) < self.duration_frames:
                padding_element = torch.zeros_like(latent_video_sample[0])
                padding = torch.stack(
                    [padding_element]
                    * (self.duration_frames - len(latent_video_sample))
                )
                latent_video_sample = torch.cat((latent_video_sample, padding), dim=0)
                assert len(latent_video_sample) == self.duration_frames, (
                    f"Video length is {len(latent_video_sample)} but should be {self.duration_frames}"
                )
            latent_video_sample = latent_video_sample.permute(1, 0, 2, 3)
            # T x C x H x W -> C x T x H x W
            output["video"] = self.transform(latent_video_sample)
            output["fps"] = target_fps
            output["padding"] = p_index
        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        if "text" in self.outputs and "EF" in row:
            # Include view in text if available
            ef_text = self.text_template.format(int(row["EF"]))
            if "View" in row:
                output["text"] = f"{ef_text}, View: {row['View']}"
            else:
                output["text"] = ef_text
        if "image" in self.outputs:
            latent_image_tensor = latent_video_tensor[
                np.random.randint(0, og_frame_count, 1)
            ][0]  # C x H x W
            output["image"] = self.transform(latent_image_tensor)

        if "key_frames" in self.outputs:
            key_frame_columns = ["Start_ED", "ES", "End_ED"]
            key_frames = []

            for col in key_frame_columns:
                if col in row and pd.notna(row[col]):
                    frame_idx = int(row[col])
                    # Ensure frame index is within bounds
                    if 0 <= frame_idx < og_frame_count:
                        key_frames.append(latent_video_tensor[frame_idx])  # C x H x W
                    else:
                        # Use first frame as fallback if index is out of bounds
                        key_frames.append(latent_video_tensor[0])
                else:
                    # Use first frame as fallback if column is missing or NaN
                    key_frames.append(latent_video_tensor[0])

            if key_frames:
                # Stack key frames channel-wise: (3 frames x C x H x W) -> (3*C x H x W)
                output["key_frames"] = torch.cat(key_frames, dim=0)
            else:
                # Fallback: use first frame repeated 3 times
                first_frame = latent_video_tensor[0]
                output["key_frames"] = torch.cat([first_frame] * 3, dim=0)

        if return_row:
            return output, row

        return output


class CardiacNet(EchoDynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__(config, split)

        # class_id
        self.class_id = config.get("class_id", "ALL")  # ASD, PAH, ALL

        # class id mapping to class names
        self.class_name_mapping = {
            0: "Atrial Septal Defect",
            1: "Non-Atrial Septal Defect",
            2: "Non-Pulmonary Arterial Hypertension",
            3: "Pulmonary Arterial Hypertension",
        }
        if self.class_id == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["class_id"] == self.class_id]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for class_id {self.class_id}")

        # Default text template for cardiac condition
        self.text_template = config.get(
            "text_template", "An echocardiography video with {} condition"
        )

    def __getitem__(self, idx):
        output, row = super().__getitem__(idx, return_row=True)

        # Initialize video reader if any video-based output is needed
        reader = None
        if "key_frames" in self.outputs or "image" in self.outputs:
            reader = decord.VideoReader(
                row["VideoPath"],
                ctx=decord.cpu(),
                width=self.resolution,
                height=self.resolution,
            )
            og_frame_count = len(reader)

        if "class_id" in self.outputs:
            output["class_id"] = row["class_id"]

        if "text" in self.outputs:
            output["text"] = self.text_template.format(
                self.class_name_mapping[row["class_id"]]
            )

        if "key_frames" in self.outputs:
            key_frame_columns = ["Start_ED", "ES", "End_ED"]
            key_frames = []

            for col in key_frame_columns:
                if col in row and pd.notna(row[col]):
                    frame_idx = int(row[col])
                    # Ensure frame index is within bounds
                    if 0 <= frame_idx < og_frame_count:
                        frame = reader.get_batch([frame_idx])[0]  # H x W x C, uint8
                        frame = frame.float() / 128.0 - 1  # normalize to [-1, 1]
                        frame = frame.permute(2, 0, 1)  # H x W x C -> C x H x W
                        key_frames.append(frame)
                    else:
                        # Use first frame as fallback if index is out of bounds
                        frame = reader.get_batch([0])[0]
                        frame = frame.float() / 128.0 - 1
                        frame = frame.permute(2, 0, 1)
                        key_frames.append(frame)
                else:
                    # Use first frame as fallback if column is missing or NaN
                    frame = reader.get_batch([0])[0]
                    frame = frame.float() / 128.0 - 1
                    frame = frame.permute(2, 0, 1)
                    key_frames.append(frame)

            if key_frames:
                # Stack key frames channel-wise: (3 frames x C x H x W) -> (3*C x H x W)
                output["key_frames"] = torch.cat(key_frames, dim=0)
            else:
                # Fallback: use first frame repeated 3 times
                first_frame = reader.get_batch([0])[0]
                first_frame = first_frame.float() / 128.0 - 1
                first_frame = first_frame.permute(2, 0, 1)
                output["key_frames"] = torch.cat([first_frame] * 3, dim=0)

        # Handle image output separately to avoid conflicts
        if "image" in self.outputs and reader is not None:
            image = reader.get_batch(np.random.randint(0, og_frame_count, 1))[
                0
            ]  # H x W x C, uint8
            image = image.float() / 128.0 - 1
            image = image.permute(2, 0, 1)  # H x W x C -> C x H x W
            output["image"] = self.transform(image)

        return output


class CardiacNetLatent(EchoDynamic):
    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        self.config = config
        super().__init__(config, split, datafolder="Latents", ext=".pt")

        self.class_id = config.get("class_id", "ALL")  # A4C, PSAX, ALL
        self.text_template = "An echocardiography video with {} condition"
        # class id mapping to class names
        self.class_name_mapping = {
            0: "Atrial Septal Defect",
            1: "Non-Atrial Septal Defect",
            2: "Non-Pulmonary Arterial Hypertension",
            3: "Pulmonary Arterial Hypertension",
        }
        if self.class_id == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["class_id"] == self.class_id]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for class_id {self.class_id}")

    def __getitem__(self, idx, return_row=False):
        row = self.metadata.iloc[idx]
        output = {
            "filename": row["FileName"],
        }

        if (
            "image" in self.outputs
            or "video" in self.outputs
            or "key_frames" in self.outputs
        ):
            latent_file = row["VideoPath"]
            latent_video_tensor = torch.load(latent_file)  # T x C x H x W
            og_fps = row["FPS"]
            og_frame_count = len(latent_video_tensor)

        if "video" in self.outputs:
            if self.target_fps == "original":
                target_fps = og_fps
            elif self.target_fps == "random":
                target_fps = np.random.randint(8, 50)
            else:
                target_fps = self.target_fps

            new_frame_count = np.floor(target_fps / og_fps * og_frame_count).astype(int)
            resample_indices = (
                np.linspace(0, og_frame_count, new_frame_count, endpoint=False)
                .round()
                .astype(int)
            )
            start_idx = (
                np.random.choice(np.arange(0, resample_indices[1]))
                if len(resample_indices) > 1 and resample_indices[1] > 1
                else 0
            )
            resample_indices = resample_indices + start_idx

            # Sample a random chunk to cover the requested duration
            start_idx = (
                np.random.choice(
                    np.arange(0, len(resample_indices) - self.duration_frames)
                )
                if len(resample_indices) > self.duration_frames
                else 0
            )
            end_idx = start_idx + self.duration_frames
            resample_indices = resample_indices[start_idx:end_idx]
            resample_indices = resample_indices[
                resample_indices < og_frame_count
            ]  # remove indices that are out of bounds

            latent_video_sample = latent_video_tensor[resample_indices]

            # Check if padding is needed
            p_index = len(latent_video_sample)
            if len(latent_video_sample) < self.duration_frames:
                padding_element = torch.zeros_like(latent_video_sample[0])
                padding = torch.stack(
                    [padding_element]
                    * (self.duration_frames - len(latent_video_sample))
                )
                latent_video_sample = torch.cat((latent_video_sample, padding), dim=0)
                assert len(latent_video_sample) == self.duration_frames, (
                    f"Video length is {len(latent_video_sample)} but should be {self.duration_frames}"
                )

            latent_video_sample = latent_video_sample.permute(
                1, 0, 2, 3
            )  # T x C x H x W -> C x T x H x W
            output["video"] = self.transform(latent_video_sample)
            output["fps"] = target_fps
            output["padding"] = p_index

        if "class_id" in self.outputs:
            output["class_id"] = row["class_id"]

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        if "text" in self.outputs:
            output["text"] = self.text_template.format(
                self.class_name_mapping[row["class_id"]]
            )

        if "image" in self.outputs:
            latent_image_tensor = latent_video_tensor[
                np.random.randint(0, og_frame_count, 1)
            ][0]  # C x H x W
            output["image"] = self.transform(latent_image_tensor)

        if "key_frames" in self.outputs:
            key_frame_columns = ["Start_ED", "ES", "End_ED"]
            key_frames = []

            for col in key_frame_columns:
                if col in row and pd.notna(row[col]):
                    frame_idx = int(row[col])
                    # Ensure frame index is within bounds
                    if 0 <= frame_idx < og_frame_count:
                        key_frames.append(latent_video_tensor[frame_idx])  # C x H x W
                    else:
                        # Use first frame as fallback if index is out of bounds
                        key_frames.append(latent_video_tensor[0])
                else:
                    # Use first frame as fallback if column is missing or NaN
                    key_frames.append(latent_video_tensor[0])

            if key_frames:
                # Stack key frames channel-wise: (3 frames x C x H x W) -> (3*C x H x W)
                output["key_frames"] = torch.cat(key_frames, dim=0)
            else:
                # Fallback: use first frame repeated 3 times
                first_frame = latent_video_tensor[0]
                output["key_frames"] = torch.cat([first_frame] * 3, dim=0)

        if return_row:
            return output, row

        return output


class CardiacNetLatentAR(Dataset):
    def __init__(
        self,
        config=None,
        datafolder="Latents",
        ext=".pt",
        target_resolution=14,
        target_fps=32,
        target_nframes=64,
        text_template="An echocardiography video with {} condition",
        outputs=["text", "prior_frame", "target_frame"],
        split=["TRAIN", "VAL", "TEST"],
        class_id="ALL",
        prior_sequence_length=64,  # Length of prior sequence
        target_sequence_length=64,  # Length of target sequence
        stride_between_sequences=32,  # Frames to skip between prior and target sequences
    ) -> None:
        # Initialize without calling _build_frame_pairs yet
        super().__init__()
        self.config = config
        self.target_fps = target_fps
        self.resolution = target_resolution
        self.outputs = outputs
        self.duration_frames = target_nframes
        self.duration_seconds = (
            self.duration_frames / self.target_fps
            if type(self.target_fps) == int
            else None
        )
        self.text_template = text_template
        self.root = config.root
        self.datafolder = datafolder
        self.ext = ext

        # 3D sequence parameters
        self.prior_sequence_length = prior_sequence_length
        self.target_sequence_length = target_sequence_length
        self.stride_between_sequences = stride_between_sequences

        self.transform = T.Compose(
            [T.Resize((self.resolution, self.resolution)), T.ToTensor()]
        )

        # LOAD DATA
        if type(split) == str:
            split = [split]
        assert [s in ["TRAIN", "VAL", "TEST"] for s in split], (
            "Splits must be a list of TRAIN, VAL, TEST"
        )
        if type(self.outputs) == str:
            self.outputs = [self.outputs]
        # print(f"Outputs: {self.root}")
        assert all(
            [o in ["text", "prior_frame", "target_frame"] for o in self.outputs]
        ), "Outputs must be a list of text, prior_frame, target_frame"
        assert os.path.exists(os.path.join(self.root, self.datafolder)), (
            f"Data folder {os.path.join(self.root, self.datafolder)} does not exist"
        )
        assert os.path.exists(os.path.join(self.root, "FileList.csv")), (
            f"FileList.csv does not exist in {self.root}"
        )

        self.metadata = pd.read_csv(os.path.join(self.root, "FileList.csv"))

        # filter by split
        self.metadata = self.metadata[self.metadata["Split"].isin(split)]
        self.len_before_filter = len(self.metadata)

        # add duration column
        self.metadata["Duration"] = (
            self.metadata["NumberOfFrames"] / self.metadata["FPS"]
        )  # won't work for pediatrics

        # filter by duration (we want videos with at least duration_frames)
        if self.duration_seconds is not None:
            self.metadata = self.metadata[
                self.metadata["Duration"] >= self.duration_seconds
            ]

        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: os.path.join(self.root, self.datafolder, x)
            if x.endswith(ext)
            else os.path.join(self.root, self.datafolder, x.split(".")[0] + ext)
        )
        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: os.path.exists(x)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]
        self.metadata.reset_index(inplace=True, drop=True)
        if len(self.metadata) == 0:
            raise ValueError(
                f"No data found in folder {os.path.join(self.root, self.datafolder)}"
            )

        # filter by view
        self.class_id = class_id  # ASD, PAH, ALL
        # class id mapping to class names
        self.class_name_mapping = {
            0: "Atrial Septal Defect",
            1: "Non-Atrial Septal Defect",
            2: "Non-Pulmonary Arterial Hypertension",
            3: "Pulmonary Arterial Hypertension",
        }
        if self.class_id == "ALL":
            pass
        else:
            self.metadata = self.metadata[self.metadata["class_id"] == self.class_id]
            self.metadata.reset_index(inplace=True, drop=True)
            if len(self.metadata) == 0:
                raise ValueError(f"No videos found for class_id {self.class_id}")

        # Build sequence pairs for latent videos
        self._build_sequence_pairs_latent()

    def _build_sequence_pairs_latent(self):
        """Build sequence pairs for 3D model training with latent videos"""
        self.sequence_pairs = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            latent_file = row["VideoPath"]
            class_id = row["class_id"]
            # class_text = self.text_template.format(int(class_id))
            class_text = self.text_template.format(self.class_name_mapping[class_id])

            try:
                latent_video_tensor = torch.load(latent_file)  # T x C x H x W
                num_frames = len(latent_video_tensor)
                # print(f"Loaded {latent_file} with shape {latent_video_tensor.shape}, frames: {num_frames}")

                # Need at least prior_sequence_length + stride + target_sequence_length frames
                min_frames_needed = (
                    self.prior_sequence_length + self.stride_between_sequences
                )  # + self.target_sequence_length

                if num_frames < min_frames_needed:
                    # print(
                    #     f"Skipping {latent_file}: not enough frames ({num_frames} < {min_frames_needed})"
                    # )
                    continue

                # Create sequence pairs
                # For each possible starting point of the prior sequence
                for start_idx in range(0, num_frames - min_frames_needed + 1):
                    prior_end_idx = start_idx + self.prior_sequence_length
                    target_start_idx = start_idx + self.stride_between_sequences
                    target_end_idx = target_start_idx + self.target_sequence_length

                    # Ensure we don't exceed the video length
                    if target_end_idx <= num_frames:
                        self.sequence_pairs.append(
                            {
                                "video_idx": idx,
                                "prior_start_idx": start_idx,
                                "prior_end_idx": prior_end_idx,
                                "target_start_idx": target_start_idx,
                                "target_end_idx": target_end_idx,
                                "class_text": class_text,
                                "class_id": class_id,
                            }
                        )
            except Exception as e:
                print(f"Error loading {latent_file}: {e}")
                continue

        print(
            f"Created {len(self.sequence_pairs)} sequence pairs from {len(self.metadata)} videos"
        )

    def __len__(self):
        return len(self.sequence_pairs)

    def __getitem__(self, idx):
        pair = self.sequence_pairs[idx]
        video_idx = pair["video_idx"]
        prior_start_idx = pair["prior_start_idx"]
        prior_end_idx = pair["prior_end_idx"]
        target_start_idx = pair["target_start_idx"]
        target_end_idx = pair["target_end_idx"]
        class_text = pair["class_text"]
        class_id = pair["class_id"]

        row = self.metadata.iloc[video_idx]
        latent_video_tensor = torch.load(row["VideoPath"])  # T x C x H x W

        # Get the sequence pairs
        prior_sequence = latent_video_tensor[
            prior_start_idx:prior_end_idx
        ]  # prior_sequence_length x C x H x W
        target_sequence = latent_video_tensor[
            target_start_idx:target_end_idx
        ]  # target_sequence_length x C x H x W

        # Rearrange to C x T x H x W format expected by 3D models
        prior_sequence = prior_sequence.permute(
            1, 0, 2, 3
        )  # C x prior_sequence_length x H x W
        target_sequence = target_sequence.permute(
            1, 0, 2, 3
        )  # C x target_sequence_length x H x W

        output = {
            "filename": row["FileName"],
            "prior_frames": prior_sequence,  # Now a sequence C x T x H x W
            "target_frames": target_sequence,  # Now a sequence C x T x H x W
            "text": class_text,
            # "class_id": torch.tensor(class_id, dtype=torch.float32),
        }
        if "class_id" in self.outputs:
            output["class_id"] = class_id

        if "text" in self.outputs and "class_id" in row:
            # Include class_name in text conditioning
            output["text"] = (
                f"{output['text']}"  # , Class: {self.class_name_mapping[row['class_id']]}
            )

        return output


class FrameFolder(Dataset):
    """config:
    - name: FrameFolder
      active: true
      params:
        video_folder: path/to/video_folders
        meta_path: path/to/FileList.csv
        outputs: ['video', 'image', 'lvef']
    """

    def __init__(self, config, split=["TRAIN", "VAL", "TEST"]) -> None:
        super().__init__()

        self.config = config
        self.video_folder = config.video_folder
        self.meta_path = config.meta_path

        self.target_nframes = config.target_nframes
        self.target_resolution = config.target_resolution

        self.metadata = pd.read_csv(self.meta_path)
        self.metadata = self.metadata[
            self.metadata["Split"].isin(split)
        ]  # filter by split

        # check if videos are reachable
        self.metadata["VideoPath"] = self.metadata["FileName"].apply(
            lambda x: os.path.join(config.video_folder, x.split(".")[0])
        )
        self.metadata["VideoExists"] = self.metadata["VideoPath"].apply(
            lambda x: (os.path.isdir(x) and len(os.listdir(x)) > 0)
        )
        self.metadata = self.metadata[self.metadata["VideoExists"]]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        output = {
            "filename": row["FileName"],
        }

        if "image" in self.outputs:
            fpath = os.path.join(row["VideoPath"])
            rand_item = np.random.choice(os.listdir(fpath))
            image = Image.open(os.path.join(fpath, rand_item))  # H x W x C, uint8
            image = np.array(image)  # H x W x C, uint8
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1).float()  # C x H x W
            image = image / 128.0 - 1  # [-1, 1]
            output["image"] = image

        if "video" in self.outputs:
            fpath = os.path.join(row["VideoPath"])
            fc = self.target_nframes
            all_frames_names = sorted(os.listdir(fpath))
            if len(all_frames_names) > fc:
                start_idx = np.random.randint(0, len(all_frames_names) - fc)
                end_idx = start_idx + fc
            else:
                start_idx = 0
                end_idx = -1
            all_frames_names = all_frames_names[start_idx:end_idx]
            all_frames_path = [os.path.join(fpath, f) for f in all_frames_names]
            all_frames = [Image.open(f) for f in all_frames_path]
            all_frames = [np.array(f) for f in all_frames]
            all_frames = np.stack(all_frames, axis=0)  # T x H x W x C
            all_frames = torch.from_numpy(all_frames)  # T x H x W x C
            all_frames = all_frames.permute(3, 0, 1, 2).float()  # C x T x H x W
            all_frames = all_frames / 128.0 - 1  # [-1, 1]

            if len(all_frames) < fc:
                padding_element = torch.zeros_like(all_frames[0])
                padding = torch.stack([padding_element] * (fc - len(all_frames)))
                all_frames = torch.cat((all_frames, padding), dim=0)
                assert len(all_frames) == fc, (
                    f"Video length is {len(all_frames)} but should be {fc}"
                )

            output["video"] = all_frames

        if "lvef" in self.outputs:
            lvef = row["EF"] / 100.0
            output["lvef"] = torch.tensor(lvef, dtype=torch.float32)

        return output


class RFBalancer(Dataset):  # Real - Fake Balancer
    """
    Balances the dataset by sampling from each dataset with equal probability.

    """

    def __init__(self, real_dataset=None, fake_dataset=None, transform=None) -> None:
        super().__init__()

        # self.datasets = [fake_dataset, real_dataset]
        self.datasets = []
        if fake_dataset is not None:
            self.datasets.append(fake_dataset)
        if real_dataset is not None:
            self.datasets.append(real_dataset)

        if len(self.datasets) == 0:
            raise ValueError("At least one dataset must be provided")

        if len(self.datasets) > 1:
            self.ds_idx = (
                np.random.rand(
                    1,
                )
                < 0.5
            )[0]  # pick the first dataset to start with
        else:
            self.ds_idx = 0

        self.ds_current = [0] * len(self.datasets)

        self.transforms = transform

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def _get_index_for_ds(self, idx):
        ds_idx = 0
        while True:
            if idx < len(self.datasets[ds_idx]):
                break
            else:
                idx -= len(self.datasets[ds_idx])
                ds_idx = (ds_idx + 1) % len(self.datasets)
        return ds_idx, idx

    def __getitem__(self, idx):
        ds_idx, idx = self._get_index_for_ds(idx)
        output = self.datasets[ds_idx][idx]  # get item from dataset
        output["real"] = float(ds_idx)  # add real/fake label

        if self.transforms is not None and "video" in output:
            output["video"] = self.transforms(output["video"])
        if self.transforms is not None and "image" in output:
            output["image"] = self.transforms(output["image"])

        return output


class ImageSet(Dataset):
    def __init__(self, root, ext=".jpg"):
        self.root = root
        self.all_images = glob(os.path.join(root, "*.jpg"))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image = Image.open(self.all_images[idx])
        image = np.array(image) / 128.0 - 1  # [0, 255] -> [-1, 1]
        image = image.transpose(2, 0, 1)  # H x W x C -> C x H x W
        return image


class TensorSet(Dataset):
    def __init__(self, root):
        self.root = root
        self.all_tensors = glob(os.path.join(root, "*.pt"))

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        tensor = torch.load(self.all_tensors[idx], map_location="cpu")
        return tensor


class TensorSetv2(Dataset):
    def __init__(self, root, num_frames=1):
        self.root = root
        self.num_frames = num_frames
        self.all_tensors = [
            path
            for path in glob(os.path.join(root, "*.pt"))
            if self._has_enough_frames(path)
        ]

    def _has_enough_frames(self, path):
        try:
            tensor = torch.load(path, map_location="cpu")
            return tensor.size(0) >= self.num_frames
        except Exception as e:
            print(f"Error loading tensor {path}: {e}")
            return False

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        tensor = torch.load(self.all_tensors[idx], map_location="cpu")
        return tensor[: self.num_frames]


class TensorSetv3(Dataset):
    """A dataset class for loading tensors from a directory and filtering them based on metadata in a CSV file.
    We use this for sampling from the original test latents."""

    def __init__(
        self,
        root,
        num_frames=1,
        ext=".pt",
        split=["TRAIN", "VAL", "TEST"],
        csv_file="FileList.csv",
    ):
        self.root = root
        self.num_frames = num_frames
        self.split = split
        self.csv_file = os.path.join(os.path.dirname(root), csv_file)
        self.ext = ext

        # Load metadata from CSV
        self.metadata = pd.read_csv(self.csv_file)
        self.metadata = self.metadata[self.metadata["Split"].isin(split)]
        self.metadata = self.metadata[self.metadata["NumberOfFrames"] >= num_frames]
        self.metadata.reset_index(inplace=True, drop=True)

        # Filter tensors based on metadata and add appropriate extensions
        self.all_tensors = []
        self.cond_values = []
        for _, row in self.metadata.iterrows():
            file_path = os.path.join(root, row["FileName"] + ext)
            if os.path.exists(file_path):
                self.all_tensors.append(file_path)
                self.cond_values.append(row["class_id"])

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        tensor = torch.load(self.all_tensors[idx], map_location="cpu")
        cond_value = self.cond_values[idx]
        # return tensor[: self.num_frames]
        return tensor[
            : self.num_frames
        ], cond_value  # Return only the first num_frames of the tensor


class TensorSetv4(Dataset):
    """A dataset class for loading tensors from a directory and filtering them based on metadata in a CSV file.
    We use this for sampling from the samples generated by the LIDM model."""

    def __init__(
        self,
        root,
        ext=".pt",
        csv_file="FileList.csv",
    ):
        self.root = root
        self.csv_file = os.path.join(os.path.dirname(root), csv_file)
        self.ext = ext

        # Load metadata from CSV
        self.metadata = pd.read_csv(self.csv_file)
        self.metadata.reset_index(inplace=True, drop=True)

        # Filter tensors based on metadata and add appropriate extensions
        self.all_tensors = []
        self.cond_values = []
        for _, row in self.metadata.iterrows():
            file_path = os.path.join(root, row["FileName"] + ext)
            if os.path.exists(file_path):
                self.all_tensors.append(file_path)
                self.cond_values.append(row["CondValue"])

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        tensor = torch.load(self.all_tensors[idx], map_location="cpu")
        cond_value = self.cond_values[idx]
        return tensor, cond_value


import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class TensorSetv5(Dataset):
    """A dataset class for loading tensors from a directory and filtering them based on metadata in a CSV file.
    We use this for sampling from the samples generated by the triplet model."""

    def __init__(
        self,
        root,
        ext=".pt",
        csv_file="FileList.csv",
    ):
        self.root = root
        self.csv_file = os.path.join(os.path.dirname(root), csv_file)
        self.ext = ext

        # Load metadata from CSV
        self.metadata = pd.read_csv(self.csv_file)
        self.metadata.reset_index(inplace=True, drop=True)

        # Modify FileName to keep only base sample name (e.g., sample_000000)
        self.metadata["FileName"] = self.metadata["FileName"].apply(
            lambda x: x.rsplit("_frame_", 1)[0]
        )

        # Drop duplicates based on modified FileName
        self.metadata = self.metadata.drop_duplicates(subset="FileName")

        # Build tensor file paths
        self.all_tensors = []
        self.cond_values = []
        for _, row in self.metadata.iterrows():
            file_path = os.path.join(self.root, row["FileName"] + ext)
            if os.path.exists(file_path):
                self.all_tensors.append(file_path)
                self.cond_values.append(row["CondValue"])
            else:
                pass  # Skip if file does not exist

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        tensor = torch.load(self.all_tensors[idx], map_location="cpu")
        cond_value = self.cond_values[idx]
        return tensor, cond_value


class SimaseUSVideoDataset(Dataset):
    def __init__(
        self,
        phase="training",
        transform=None,
        latents_csv="./",
        training_latents_base_path="./",
        in_memory=True,
        generator_seed=None,
    ):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        PHASE_TO_SPLIT = {"training": "TRAIN", "validation": "VAL", "testing": "TEST"}
        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(
            drop=True
        )
        self.transform = transform

        if generator_seed is None:
            self.generator = np.random.default_rng()
            # unseeded
        else:
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory:
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid_a = self.get_vid(index)
        if self.transform is not None:
            vid_a = self.transform(vid_a)
        return vid_a

    def reset_generator(self):
        self.generator = np.random.default_rng(self.generator_seed)

    def get_vid(self, index, from_disk=False):
        if self.in_memory and not from_disk:
            return self.videos[index]
        else:
            path = self.df.iloc[index]["FileName"].split(".")[0] + ".pt"
            path = os.path.join(self.training_latents_base_path, path)
            return torch.load(path)

    def load_videos(self):
        self.videos = []
        print("Preloading videos")
        for i in range(len(self)):
            self.videos.append(self.get_vid(i, from_disk=True))


class SiameseUSDataset(Dataset):
    def __init__(
        self,
        phase="training",
        transform=None,
        latents_csv="./",
        training_latents_base_path="./",
        in_memory=True,
        generator_seed=None,
    ):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        PHASE_TO_SPLIT = {"training": "TRAIN", "validation": "VAL", "testing": "TEST"}
        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(
            drop=True
        )

        self.transform = transform

        if generator_seed is None:
            self.generator = np.random.default_rng()
            # unseeded
        else:
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory:
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid_a = torch.clone(self.get_vid(index))
        if self.generator.uniform() < 0.5:
            vid_b = torch.clone(
                self.get_vid(
                    (index + self.generator.integers(low=1, high=len(self))) % len(self)
                )
            )  # random different vid
            y = 0.0
        else:
            vid_b = torch.clone(vid_a)
            y = 1.0

        if self.transform is not None:
            vid_a = self.transform(vid_a)
            vid_b = self.transform(vid_b)

        frame_a = self.generator.integers(len(vid_a))
        frame_b = (frame_a + self.generator.integers(low=1, high=len(vid_b))) % len(
            vid_b
        )
        # print(f"Dataloader: framea {frame_a} - frame_b {frame_b} - y: {y}")
        return vid_a[frame_a], vid_b[frame_b], y

    def reset_generator(self):
        self.generator = np.random.default_rng(self.generator_seed)

    def get_vid(self, index, from_disk=False):
        if self.in_memory and not from_disk:
            return self.videos[index]
        else:
            path = self.df.iloc[index]["FileName"].split(".")[0] + ".pt"
            path = os.path.join(self.training_latents_base_path, path)
            return torch.load(path)

    def load_videos(self):
        self.videos = []
        print("Preloading videos")
        for i in range(len(self)):
            self.videos.append(self.get_vid(i, from_disk=True))


def instantiate_dataset(configs, split=["TRAIN", "VAL", "TEST"]):
    # config = config.copy()
    # assert config.get("datasets", False), "No 'datasets' key found in config"

    # Check if number of frames and resolution are the same for all datasets
    target_nframes = None
    target_resolution = None
    reference_name = None
    for dataset_config in configs:
        if dataset_config.get("active", True):
            if reference_name is None:
                reference_name = dataset_config.name
            if target_nframes is None:
                target_nframes = dataset_config.params.target_nframes
            else:
                newd = dataset_config.params.target_nframes
                assert newd == target_nframes, (
                    f"All datasets must ouput the same number of frames, got {reference_name}: {target_nframes} frames and {dataset_config.name}: {newd} frames."
                )
            if target_resolution is None:
                target_resolution = dataset_config.params.target_resolution
            else:
                assert dataset_config.params.target_resolution == target_resolution, (
                    f"All datasets must have the same target_resolution, got {reference_name}: {target_resolution} and {dataset_config.name}: {dataset_config.params.target_resolution}."
                )

    datasets = []
    for dataset_config in configs:
        if dataset_config.get("active", True):
            datasets.append(
                globals()[dataset_config.name](dataset_config.params, split=split)
            )

    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)

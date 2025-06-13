import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple, Union


class VideoDataset(Dataset):
    """
    Dataset for video classification.
    Handles both real and synthetic data based on configuration.
    """

    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        split: str = "TRAIN",
        transform=None,
        frames_per_clip: int = 64,
        frame_sampling: str = "uniform",
        sampling_rate: int = 1,
        synthetic_csv_path: Optional[str] = None,
        synthetic_data_dir: Optional[str] = None,
        use_synthetic_for_split: Optional[List[str]] = None,
        selected_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to the CSV file with video metadata
            data_dir: Directory containing the video frames
            split: Data split to use ('train', 'val', or 'test')
            transform: Optional transform to apply to the frames
            frames_per_clip: Number of frames to sample from each video
            frame_sampling: Method for sampling frames ('uniform', 'random')
            sampling_rate: Take every Nth frame (default=1 for all frames)
            synthetic_csv_path: Path to the CSV file with synthetic video metadata
            synthetic_data_dir: Directory containing the synthetic video frames
            use_synthetic_for_split: Which splits should use synthetic data ('train', 'val', 'test')
            selected_classes: List of class names to include (if None, include all classes)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.frames_per_clip = frames_per_clip
        self.frame_sampling = frame_sampling
        self.sampling_rate = sampling_rate
        self.selected_classes = selected_classes

        # Set up synthetic data configuration
        self.use_synthetic = (
            synthetic_csv_path is not None and synthetic_data_dir is not None
        )
        self.synthetic_data_dir = synthetic_data_dir
        self.use_synthetic_for_split = use_synthetic_for_split or []

        # Determine if this split should use synthetic data
        self.use_synthetic_for_current_split = (
            self.use_synthetic and split in self.use_synthetic_for_split
        )

        # Load CSV data
        df = pd.read_csv(csv_path)
        self.df = df[df["Split"] == split].reset_index(drop=True)

        if self.use_synthetic and self.use_synthetic_for_current_split:
            synthetic_df = pd.read_csv(synthetic_csv_path)
            synthetic_df = synthetic_df[synthetic_df["Split"] == split].reset_index(
                drop=True
            )
            self.df = synthetic_df

        # Filter by selected classes if provided
        if self.selected_classes is not None:
            # Handle case where selected_classes is a list containing a comma-separated string
            if len(self.selected_classes) == 1 and "," in self.selected_classes[0]:
                self.selected_classes = self.selected_classes[0].split(",")

            # Try to convert string representations of integers to integers if needed
            class_filters = []
            for cls in self.selected_classes:
                try:
                    # If the class ID is a string representation of a number, convert it
                    class_filters.append(int(cls))
                except (ValueError, TypeError):
                    # If it's not convertible to int, keep it as is
                    class_filters.append(cls)
            # Filter the dataframe by class_id
            self.df = self.df[self.df["class_id"].isin(class_filters)].reset_index(
                drop=True
            )
            if len(self.df) == 0:
                raise ValueError(
                    f"No samples found for selected classes {self.selected_classes} in split {split}. "
                    f"Available classes are: {sorted(set(df[df['Split'] == split]['class_id'].tolist()))}"
                )

        # Create class to index mapping
        self.classes = sorted(self.df["class_name"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Calculate class weights
        self._calculate_class_weights()

    def _calculate_class_weights(self):
        """
        Calculate class weights based on the frequency of each class in the dataset.
        The weights are inversely proportional to the class frequencies.
        """
        class_counts = self.df["class_name"].value_counts().to_dict()
        total_samples = len(self.df)

        # Get counts in the same order as self.classes
        counts = np.array([class_counts.get(cls, 0) for cls in self.classes])

        # Prevent division by zero
        counts = np.where(counts == 0, 1, counts)

        # Calculate weights (inverse frequency)
        weights = total_samples / (len(self.classes) * counts)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        # Convert to tensor
        self.class_weights = torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_data = self.df.iloc[idx]
        video_filename = video_data["FileName"]
        class_name = video_data["class_name"]
        num_frames = video_data["NumberOfFrames"]

        # Determine data directory based on configuration
        base_dir = (
            self.synthetic_data_dir
            if self.use_synthetic_for_current_split
            else self.data_dir
        )
        video_dir = os.path.join(base_dir, video_filename)

        # Sample frames
        frames = self._sample_frames(video_dir, num_frames)

        # Apply transformations
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)

        # Get class index
        class_idx = self.class_to_idx[class_name]

        return frames, class_idx

    def _sample_frames(self, video_dir, num_frames):
        """
        Sample frames from the video.

        Args:
            video_dir: Directory containing the video frames
            num_frames: Total number of frames in the video

        Returns:
            List of sampled frames as PIL Images
        """
        available_frames = sorted(
            [f for f in os.listdir(video_dir) if f.endswith((".jpg", ".png"))]
        )
        # available_frames = sorted(
        #     [f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png'))],
        #     key=lambda x: int(os.path.splitext(x)[0])
        # )

        # Apply sampling rate by selecting every nth frame
        if hasattr(self, "sampling_rate") and self.sampling_rate > 1:
            available_frames = available_frames[:: self.sampling_rate]
            num_frames = len(available_frames)

        if num_frames < self.frames_per_clip:
            # If we have fewer frames than needed, duplicate frames
            frame_indices = np.linspace(
                0, num_frames - 1, self.frames_per_clip, dtype=int
            )
        else:
            if self.frame_sampling == "uniform":
                # Uniform sampling
                frame_indices = np.linspace(
                    0, num_frames - 1, self.frames_per_clip, dtype=int
                )
            elif self.frame_sampling == "random":
                # Random sampling
                if self.split == "TRAIN":
                    frame_indices = sorted(
                        np.random.choice(
                            num_frames,
                            self.frames_per_clip,
                            replace=num_frames < self.frames_per_clip,
                        )
                    )
                else:
                    # For val/test, use uniform sampling for reproducibility
                    frame_indices = np.linspace(
                        0, num_frames - 1, self.frames_per_clip, dtype=int
                    )
            else:
                raise ValueError(
                    f"Unsupported frame sampling method: {self.frame_sampling}"
                )

        # Load sampled frames
        frames = []
        for idx in frame_indices:
            if idx >= len(available_frames):
                idx = len(available_frames) - 1
            frame_path = os.path.join(video_dir, available_frames[idx])
            try:
                frame = Image.open(frame_path).convert("RGB")
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                # If loading fails, create a black frame
                frame = Image.new("RGB", (112, 112), (0, 0, 0))
                frames.append(frame)

        return frames


def get_video_transforms(
    split, input_size=112, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    """
    Get transforms for video frames.

    Args:
        split: Dataset split ('train', 'val', or 'test')
        input_size: Input frame size
        mean: Normalization mean
        std: Normalization standard deviation

    Returns:
        Composed transforms
    """
    if split == "train":
        return transforms.Compose(
            [
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                # transforms.Resize(input_size + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def create_video_dataloaders(
    config: Dict,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for video classification.

    Args:
        config: Configuration dictionary with the following keys:
            - csv_path: Path to the CSV file with video metadata
            - data_dir: Directory containing the video frames
            - batch_size: Batch size for dataloaders
            - num_workers: Number of workers for dataloaders
            - frames_per_clip: Number of frames to sample per video
            - frame_sampling: Method for sampling frames
            - sampling_rate: Take every Nth frame (default=1 for all frames)
            - synthetic_csv_path: Path to CSV file with synthetic data (optional)
            - synthetic_data_dir: Directory with synthetic data (optional)
            - use_synthetic_for: Which splits should use synthetic data ('train', 'val', 'test')
            - pin_memory: Whether to use pin_memory for DataLoader (default: False)

    Returns:
        Dictionary with train, val, and test dataloaders
    """
    # Extract configuration
    csv_path = config["csv_path"]
    data_dir = config["data_dir"]
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    frames_per_clip = config.get("frames_per_clip", 16)
    frame_sampling = config.get("frame_sampling", "uniform")
    sampling_rate = config.get("sampling_rate", 1)
    selected_classes = config.get("selected_classes", None)
    synthetic_csv_path = config.get("synthetic_csv_path")
    synthetic_data_dir = config.get("synthetic_data_dir")
    use_synthetic_for = config.get("use_synthetic_for", [])
    pin_memory = config.get("pin_memory", False)

    # Create transforms
    train_transform = get_video_transforms("train")
    val_transform = get_video_transforms("val")
    test_transform = get_video_transforms("test")

    # Create datasets
    train_dataset = VideoDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        split="TRAIN",
        transform=train_transform,
        frames_per_clip=frames_per_clip,
        frame_sampling=frame_sampling,
        sampling_rate=sampling_rate,
        synthetic_csv_path=synthetic_csv_path,
        synthetic_data_dir=synthetic_data_dir,
        use_synthetic_for_split=use_synthetic_for,
        selected_classes=selected_classes,
    )

    val_dataset = VideoDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        split="VAL",
        transform=val_transform,
        frames_per_clip=frames_per_clip,
        frame_sampling=frame_sampling,
        sampling_rate=sampling_rate,
        synthetic_csv_path=synthetic_csv_path,
        synthetic_data_dir=synthetic_data_dir,
        use_synthetic_for_split=use_synthetic_for,
        selected_classes=selected_classes,
    )

    test_dataset = VideoDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        split="TEST",
        transform=test_transform,
        frames_per_clip=frames_per_clip,
        frame_sampling=frame_sampling,
        sampling_rate=sampling_rate,
        synthetic_csv_path=synthetic_csv_path,
        synthetic_data_dir=synthetic_data_dir,
        use_synthetic_for_split=use_synthetic_for,
        selected_classes=selected_classes,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,  # Use the configured value
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,  # Use the configured value
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,  # Use the configured value
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

import argparse
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from echo.classification.model import get_video_classifier
from echo.classification.data import create_video_dataloaders, VideoDataset
from torch.utils.data import DataLoader


def load_model(backbone, checkpoint_path, device):
    """
    Load a trained video classification model from a checkpoint

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on

    Returns:
        Loaded model and list of class names
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get classes
    classes = checkpoint.get("classes", None)
    if classes is None:
        raise ValueError("No class information found in checkpoint")

    num_classes = len(classes)
    print(f"Found {num_classes} classes in checkpoint")

    # Extract model info
    state_dict = checkpoint["model_state_dict"]

    if backbone == "resnet50" or backbone == "resnet43":
        if any("attention" in key for key in state_dict.keys()):
            pool_type = "attention"
        else:
            pool_type = "avg"  # Default pooling
        model = get_video_classifier(
            num_classes=num_classes, backbone=backbone, pool_type=pool_type
        )
    elif backbone == "r3d_18" or backbone == "r2plus1d_18":
        model = get_video_classifier(num_classes=num_classes, backbone=backbone)
    else:
        raise ValueError("Invalid model  backbone!")

    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, classes


def predict_videos(model, dataloader, device, top_k=1):
    """
    Run predictions on videos using the trained model

    Args:
        model: Trained model
        dataloader: DataLoader with videos to predict
        device: Device to run inference on
        top_k: Number of top probabilities to return

    Returns:
        Tuple of (all_predictions, all_video_ids, all_probs)
    """
    all_predictions = []
    all_video_ids = []
    all_probs = []

    # Set model to evaluation mode
    model.eval()

    # Get video IDs from dataset
    try:
        # Try to access the DataFrame directly from the dataset
        if (
            hasattr(dataloader.dataset, "df")
            and "FileName" in dataloader.dataset.df.columns
        ):
            # Use FileName as video_id
            video_ids = dataloader.dataset.df["FileName"].tolist()
            print(f"Using {len(video_ids)} FileName entries as video IDs")
        elif (
            hasattr(dataloader.dataset, "df")
            and "video_id" in dataloader.dataset.df.columns
        ):
            # Use video_id column if available
            video_ids = dataloader.dataset.df["video_id"].tolist()
            print(f"Using {len(video_ids)} video_id entries from DataFrame")
        else:
            # Fallback to generating sequential IDs
            print("No video IDs found in dataset, using sequential IDs")
            video_ids = [f"video_{i}" for i in range(len(dataloader.dataset))]
    except Exception as e:
        print(f"Error extracting video IDs: {e}")
        print("Using sequential IDs instead")
        video_ids = [f"video_{i}" for i in range(len(dataloader.dataset))]

    print(f"Total videos to process: {len(video_ids)}")

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Predicting")):
            # Calculate the indices for this batch
            batch_size = inputs.size(0)
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + batch_size, len(dataloader.dataset))

            # Get video IDs for this batch
            batch_video_ids = video_ids[start_idx:end_idx]
            all_video_ids.extend(batch_video_ids)

            # Move inputs to device
            inputs = inputs.to(device)

            # Get predictions
            outputs = model(inputs)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Get top predictions and their indices
            values, indices = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)

            # Store predictions (still using argmax for the main prediction)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())

            # Store top-k probabilities and their indices
            all_probs.extend(zip(indices.cpu().numpy(), values.cpu().numpy()))

    return all_predictions, all_video_ids, all_probs


def update_csv_with_predictions(
    csv_path, video_ids, predictions, classes, probs=None, output_csv=None, top_k=3
):
    """
    Update the input CSV with predicted class labels and probabilities

    Args:
        csv_path: Path to the input CSV file
        video_ids: List of video IDs that were predicted
        predictions: List of prediction indices
        classes: List of class names
        probs: List of tuples containing (indices, probabilities)
        output_csv: Path to save the updated CSV (if None, will modify with _predictions suffix)
        top_k: Number of top probabilities to save

    Returns:
        Path to the updated CSV
    """
    # Read the input CSV
    df = pd.read_csv(csv_path)

    # Create mappings from video_id to predicted class and class_id
    pred_dict_class = {vid: classes[pred] for vid, pred in zip(video_ids, predictions)}
    pred_dict_class_id = {vid: int(pred) for vid, pred in zip(video_ids, predictions)}

    # Check if 'video_id' column exists, if not use 'FileName'
    id_column = "video_id" if "video_id" in df.columns else "FileName"
    print(f"Using '{id_column}' column to match predictions to videos")

    # Add predictions to the DataFrame
    df["predicted_class"] = df[id_column].map(pred_dict_class)
    df["predicted_class_id"] = df[id_column].map(pred_dict_class_id)

    # Add probability information if available
    if probs is not None:
        # Create a dictionary to map video IDs to probability info
        prob_dict = {vid: prob_data for vid, prob_data in zip(video_ids, probs)}

        # Function to extract probability for the predicted class
        def get_main_prob(video_id):
            if video_id not in prob_dict:
                return None
            indices, values = prob_dict[video_id]
            pred_idx = predictions[video_ids.index(video_id)]
            # Find position of prediction in top-k indices
            pos = np.where(indices == pred_idx)[0]
            if len(pos) > 0:
                return float(values[pos[0]])
            return None

        # Add column for the probability of the predicted class
        df["prediction_probability"] = df[id_column].apply(get_main_prob)

        # Add columns for top-k predictions and probabilities
        for k in range(min(top_k, len(classes))):
            # Function to get the k-th class
            def get_kth_class(video_id, k_idx):
                if video_id not in prob_dict:
                    return None, None
                indices, values = prob_dict[video_id]
                if k_idx < len(indices):
                    return classes[indices[k_idx]], float(values[k_idx])
                return None, None

            # Add columns for k-th class and probability
            df[f"top_{k + 1}_class"] = df[id_column].apply(
                lambda vid: get_kth_class(vid, k)[0]
            )
            df[f"top_{k + 1}_probability"] = df[id_column].apply(
                lambda vid: get_kth_class(vid, k)[1]
            )

    # Handle videos without predictions (if any)
    missing_preds = df[df["predicted_class"].isna()][id_column].count()
    if missing_preds > 0:
        print(f"Warning: {missing_preds} videos in CSV did not receive predictions")

    # Determine output path
    if output_csv is None:
        base, ext = os.path.splitext(csv_path)
        output_csv = f"{base}_predictions{ext}"

    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

    return output_csv


def create_prediction_dataloader(config, split="TEST"):
    """
    Create a dataloader for prediction for a specific split

    Args:
        config: Configuration dictionary
        split: Data split to use ('TRAIN', 'VAL', or 'TEST')

    Returns:
        DataLoader configured for prediction
    """
    # Read CSV to get class information
    df = pd.read_csv(config["csv_path"])

    # Filter by split
    df = df[df["Split"] == split].reset_index(drop=True)

    # Filter by selected classes if specified
    if config.get("selected_classes") is not None:
        selected_classes = config["selected_classes"]
        # Try to convert string representations of integers to integers if needed
        class_filters = []
        for cls in selected_classes:
            try:
                # If the class ID is a string representation of a number, convert it
                class_filters.append(int(cls))
            except (ValueError, TypeError):
                # If it's not convertible to int, keep it as is
                class_filters.append(cls)

        # Filter the dataframe by class_id
        df = df[df["class_id"].isin(class_filters)].reset_index(drop=True)

    if len(df) == 0:
        return None, None

    # Save filtered dataframe to a temporary CSV
    temp_csv_path = f"{os.path.splitext(config['csv_path'])[0]}_temp_{split}.csv"
    df.to_csv(temp_csv_path, index=False)

    # Create dataset using the temporary CSV
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VideoDataset(
        csv_path=temp_csv_path,
        data_dir=config["data_dir"],
        split=split,
        transform=transform,
        frames_per_clip=config["frames_per_clip"],
        frame_sampling=config["frame_sampling"],
        sampling_rate=config["sampling_rate"],
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
    )

    return dataloader, temp_csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict video classes using a trained model"
    )

    # Required arguments
    parser.add_argument("--backbone", type=str, required=True, help="Model type")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file with video metadata",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the video frames",
    )

    # Optional arguments
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save the output CSV with predictions",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for prediction"
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=64,
        help="Number of frames to sample per video",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=1, help="Sampling rate for frames"
    )
    parser.add_argument(
        "--frame-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
        help="Method for sampling frames",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )

    # New arguments for filtering
    parser.add_argument(
        "--selected-classes",
        type=str,
        nargs="+",
        default=None,
        help="List of classes to predict on (class_id values)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["TEST"],
        choices=["TRAIN", "VAL", "TEST", "ALL"],
        help="Which splits to predict on",
    )

    # New argument for top-k probabilities
    parser.add_argument(
        "--top-k", type=int, default=1, help="Number of top class probabilities to save"
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    model, classes = load_model(args.backbone, args.checkpoint, device)
    print(f"Classes: {classes}")

    # Determine which splits to process
    splits_to_process = (
        ["TRAIN", "VAL", "TEST"] if "ALL" in args.splits else args.splits
    )

    # Create dataloader configuration
    dataloader_config = {
        "csv_path": args.csv_path,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "frames_per_clip": args.frames_per_clip,
        "frame_sampling": args.frame_sampling,
        "sampling_rate": args.sampling_rate,
        "selected_classes": args.selected_classes,
    }

    # Create combined output dataframe
    combined_df = pd.DataFrame()
    temp_csvs = []

    # Process each split
    for split in splits_to_process:
        print(f"\nProcessing {split} split...")

        # Create dataloader for this split
        dataloader, temp_csv_path = create_prediction_dataloader(
            dataloader_config, split
        )

        if temp_csv_path:
            temp_csvs.append(temp_csv_path)

        if dataloader is None:
            print(f"No videos to predict for {split} split with the specified filters.")
            continue

        print(f"Found {len(dataloader.dataset)} videos for {split} split")

        # Run prediction
        start_time = time.time()
        predictions, video_ids, probs = predict_videos(
            model, dataloader, device, top_k=args.top_k
        )
        elapsed_time = time.time() - start_time
        print(f"Prediction completed in {elapsed_time:.2f} seconds")

        # Update CSV with predictions
        if args.output_csv:
            output_csv = f"{os.path.splitext(args.output_csv)[0]}_{split}.csv"
        else:
            base, ext = os.path.splitext(args.csv_path)
            output_csv = f"{base}_{split}_predictions{ext}"

        updated_csv = update_csv_with_predictions(
            temp_csv_path,
            video_ids,
            predictions,
            classes,
            probs=probs,
            output_csv=output_csv,
            top_k=args.top_k,
        )

        print(f"Updated CSV saved to {updated_csv}")

        # Add to combined dataframe
        split_df = pd.read_csv(updated_csv)
        split_df["Split"] = split  # Ensure Split column is preserved
        combined_df = pd.concat([combined_df, split_df], ignore_index=True)

    # Save combined predictions
    if not combined_df.empty:
        if args.output_csv:
            combined_output = args.output_csv
        else:
            base, ext = os.path.splitext(args.csv_path)
            combined_output = f"{base}_all_predictions{ext}"

        combined_df.to_csv(combined_output, index=False)
        print(f"\nCombined predictions for all splits saved to {combined_output}")

    # Clean up temporary files
    for temp_csv in temp_csvs:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    print("Finished prediction process.")


if __name__ == "__main__":
    main()

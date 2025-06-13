#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_and_update_filelist(
    csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42
):
    """
    python scripts/2split_and_update_filelist.py --csv_path /nfs/usrhome/khmuhammad/Echodream/datasets/CardiacNet/FileList.csv
    Split the file list into training, validation, and test sets and update the CSV.

    Parameters:
    -----------
    csv_path : str
        Path to the FileList.csv file
    train_ratio : float
        Ratio of samples for training set
    val_ratio : float
        Ratio of samples for validation set
    test_ratio : float
        Ratio of samples for test set
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Check if ratios sum to 1
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-10:
            print("Error: Train, validation, and test ratios must sum to 1")
            return False

        # Read the CSV file
        print(f"Reading file list from {csv_path}")
        df = pd.read_csv(csv_path)

        # Make sure we have data
        if len(df) == 0:
            print("Error: No data found in the CSV file")
            return False

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Group patients by class to enable stratified splitting
        class_name_col = "class_name"
        if (
            "full_class" in df.columns
        ):  # Use full_class if available (for multi-condition datasets)
            class_name_col = "full_class"

        # Create a dataframe with patient_id and class
        patient_class_df = df.drop_duplicates(subset=["patient_id", class_name_col])
        patient_classes = {}

        # Dictionary to store patients by class
        class_patients = defaultdict(list)

        for _, row in patient_class_df.iterrows():
            patient_id = row["patient_id"]
            class_name = row[class_name_col]
            patient_classes[patient_id] = class_name
            class_patients[class_name].append(patient_id)

        # Check if we have enough patients in each class
        for class_name, patients in class_patients.items():
            if len(patients) < 3:
                print(
                    f"Warning: Class {class_name} has fewer than 3 patients ({len(patients)}). Split might not be optimal."
                )

        # Create a dictionary to store which split each patient belongs to
        patient_splits = {}

        # For each class, split the patients stratified by class
        for class_name, patients in class_patients.items():
            # Convert to numpy array for splitting
            patients_array = np.array(patients)
            np.random.shuffle(patients_array)

            # Calculate split sizes
            train_size = int(len(patients_array) * train_ratio)
            val_size = int(len(patients_array) * val_ratio)

            # Perform the splits
            train_patients = patients_array[:train_size]
            val_patients = patients_array[train_size : train_size + val_size]
            test_patients = patients_array[train_size + val_size :]

            # Assign splits to patients
            for p in train_patients:
                patient_splits[p] = "TRAIN"
            for p in val_patients:
                patient_splits[p] = "VAL"
            for p in test_patients:
                patient_splits[p] = "TEST"

        # Now assign splits to the dataframe based on patient_id
        df["Split"] = df["patient_id"].map(patient_splits)

        # Print split statistics
        print(f"Split statistics:")
        print(f"  Total samples: {len(df)}")

        # Print statistics for each split
        for split in ["TRAIN", "VAL", "TEST"]:
            split_df = df[df["Split"] == split]
            class_counts = split_df[class_name_col].value_counts().to_dict()
            print(
                f"  {split}: {len(split_df)} samples, {len(split_df['patient_id'].unique())} patients"
            )
            print(f"    Classes: {class_counts}")

            # Calculate and print class proportions
            class_props = {
                cls: count / len(split_df) for cls, count in class_counts.items()
            }
            print(f"    Class proportions: {class_props}")

        # Save the updated CSV file
        df.to_csv(csv_path, index=False)
        print(f"Updated {csv_path} with train/val/test splits")
        return True

    except Exception as e:
        print(f"Error updating splits: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Split file list into train, validation, and test sets"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to the FileList.csv file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of samples for training set (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of samples for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio of samples for test set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.isfile(args.csv_path):
        print(f"Error: File {args.csv_path} does not exist")
        return

    # Split and update the file list
    success = split_and_update_filelist(
        args.csv_path, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    if success:
        print("Split operation completed successfully")
    else:
        print("Split operation failed")


if __name__ == "__main__":
    main()

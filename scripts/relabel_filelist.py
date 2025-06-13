import pandas as pd
import argparse
import os

"""
python scripts/relabel_filelist.py --output /nfs/usrhome/khmuhammad/Echonet/datasets/EchoNet-Synthetic/arlvdm_dynamic_with_lvdm_released_s64/FileList_new.csv 

"""


def update_ef_scores(
    filelist_path,
    train_path,
    val_path,
    test_path,
    file_col="FileName",
    ef_col="EF",
    output_path=None,
):
    """
    Update EF scores in FileList.csv based on values from train.csv, val.csv, and test.csv
    """
    # Read all CSV files
    filelist_df = pd.read_csv(filelist_path)
    train_df = pd.read_csv(train_path, header=None)
    val_df = pd.read_csv(val_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Create a mapping of filenames to their updated EF scores
    ef_mapping = {}

    for df in [train_df, val_df, test_df]:
        for _, row in df.iterrows():
            filename_without_ext = os.path.splitext(row[0])[0]  # Remove file extension
            ef_mapping[filename_without_ext] = row[
                1
            ]  # Assuming FileName is the first column and EF is the second

    # Update the EF scores in FileList
    updated_count = 0
    for idx, row in filelist_df.iterrows():
        if row[file_col] in ef_mapping:
            filelist_df.at[idx, ef_col] = ef_mapping[row[file_col]]
            updated_count += 1
    # Save the updated DataFrame
    if output_path is None:
        output_path = filelist_path

    filelist_df.to_csv(output_path, index=False)
    print(f"Updated {updated_count} EF scores out of {len(filelist_df)} entries")
    print(f"Updated FileList saved to {output_path}")

    return filelist_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update EF scores in FileList.csv")
    parser.add_argument(
        "--filelist",
        default="/nfs/usrhome/khmuhammad/Echonet/datasets/EchoNet-Synthetic/arlvdm_dynamic_with_lvdm_released_s64/FileList.csv",
        help="Path to FileList.csv",
    )
    parser.add_argument(
        "--train",
        default="/nfs/usrhome/khmuhammad/Echonet/experiments/relabel_arlvdm_s64/train_predictions.csv",
        help="Path to train.csv",
    )
    parser.add_argument(
        "--val",
        default="/nfs/usrhome/khmuhammad/Echonet/experiments/relabel_arlvdm_s64/val_predictions.csv",
        help="Path to val.csv",
    )
    parser.add_argument(
        "--test",
        default="/nfs/usrhome/khmuhammad/Echonet/experiments/relabel_arlvdm_s64/test_predictions.csv",
        help="Path to test.csv",
    )
    parser.add_argument(
        "--output",
        help="Path to save the updated FileList (defaults to overwriting original)",
    )
    parser.add_argument(
        "--file-col", default="FileName", help="Name of the file column"
    )
    parser.add_argument("--ef-col", default="EF", help="Name of the EF column")

    args = parser.parse_args()

    update_ef_scores(
        args.filelist,
        args.train,
        args.val,
        args.test,
        file_col=args.file_col,
        ef_col=args.ef_col,
        output_path=args.output,
    )

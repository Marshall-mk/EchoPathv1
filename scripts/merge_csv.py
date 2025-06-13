import pandas as pd

# List of CSV files (adjust paths as needed)
csv_files = [
    "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv1/FileList.csv",
    "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv2/FileList.csv",
    # "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddpm_csv3/FileList.csv",
    # "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddpm_csv4/FileList.csv"
]

# Initialize list to store dataframes and track cumulative samples
dfs = []
starting_index = 0

# Process each CSV file
for i, csv_file in enumerate(csv_files):
    # Read the CSV
    df = pd.read_csv(csv_file)

    # Extract the numeric index from "FileName" (e.g., "000000" from "sample_000000")
    df["original_index"] = df["FileName"].apply(lambda x: int(x.split("_")[1]))

    # Determine the maximum index and number of samples
    max_index = df["original_index"].max()
    num_samples = max_index + 1  # Assuming consecutive indices from 0

    # Set the offset (starting index for this CSV)
    offset = starting_index if i > 0 else 0

    # Update "FileName" with the new index
    df["FileName"] = df["original_index"].apply(lambda x: f"sample_{x + offset:06d}")

    # Drop the temporary 'original_index' column
    df.drop("original_index", axis=1, inplace=True)

    # Add the updated dataframe to the list
    dfs.append(df)

    # Update the starting index for the next CSV
    starting_index += num_samples

# Merge all dataframes into one
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged CSV
merged_df.to_csv(
    "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv/FileList.csv",
    index=False,
)

print(
    "CSV files merged successfully into '/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv/FileList.csv'."
)

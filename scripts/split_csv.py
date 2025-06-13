import pandas as pd


def split_csv_pandas(input_file, output_file1, output_file2, split_ratio=0.5):
    # Read the entire CSV file
    df = pd.read_csv(input_file)

    # Calculate split point
    split_point = int(len(df) * split_ratio)

    # Split the dataframe
    df1 = df.iloc[:split_point]
    df2 = df.iloc[split_point:]

    # Write to output files
    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)


# Usage example:
split_csv_pandas(
    "/home/khmuhammad/Echonet/FileList1.csv",
    "/home/khmuhammad/Echonet/FileList1.csv",
    "/home/khmuhammad/Echonet/FileList2.csv",
    0.5,
)

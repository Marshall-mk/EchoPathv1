import pandas as pd
import argparse

# Step 1: Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Merge two CSV files based on prediction probabilities."
)
parser.add_argument(
    "--csv_a",
    required=True,
    help="Path to the first CSV file (e.g., output_predictions_pah.csv).",
)
parser.add_argument(
    "--csv_b",
    required=True,
    help="Path to the second CSV file (e.g., output_predictions_asd.csv).",
)
parser.add_argument(
    "--output", required=True, help="Path to save the combined CSV file."
)
args = parser.parse_args()

# Step 2: Load the CSV files
csv_a = pd.read_csv(args.csv_a)  # CSV A
csv_b = pd.read_csv(args.csv_b)  # CSV B

# Step 3: Set the sample identifier as the index
# Assuming "FileName" is the column identifying samples
csv_a.set_index("FileName", inplace=True)
csv_b.set_index("FileName", inplace=True)

# Step 4: Find samples common to both files
common_samples = csv_a.index.intersection(csv_b.index)
class_name_mapping = {
    0: "Atrial Septal Defect",
    1: "Non-Atrial Septal Defect",
    2: "Non-Pulmonary Arterial Hypertension",
    3: "Pulmonary Arterial Hypertension",
}

# Step 5: Compare probabilities and select the higher one
combined_data = []
for sample in common_samples:
    prob_a = csv_a.loc[sample, "prediction_probability"]
    prob_b = csv_b.loc[sample, "prediction_probability"]

    if prob_a > prob_b:
        selected_class_id = csv_a.loc[sample, "predicted_class_id"]
        selected_prob = prob_a
        source = "CSV_A"
    else:
        selected_class_id = csv_b.loc[sample, "predicted_class_id"]
        selected_prob = prob_b
        source = "CSV_B"

    # Collect the data for the new CSV
    combined_data.append(
        {
            "FileName": sample,
            "NumberOfFrames": csv_a.loc[sample, "NumberOfFrames"]
            if "NumberOfFrames" in csv_a.columns
            else csv_b.loc[sample, "NumberOfFrames"],
            "FPS": csv_a.loc[sample, "FPS"]
            if "FPS" in csv_a.columns
            else csv_b.loc[sample, "FPS"],
            "FrameWidth": csv_a.loc[sample, "FrameWidth"]
            if "FrameWidth" in csv_a.columns
            else csv_b.loc[sample, "FrameWidth"],
            "FrameHeight": csv_a.loc[sample, "FrameHeight"]
            if "FrameHeight" in csv_a.columns
            else csv_b.loc[sample, "FrameHeight"],
            "class_id": selected_class_id,
            "probability": selected_prob,
            "source": source,
            "class_name": class_name_mapping.get(selected_class_id, "Unknown"),
            "Split": csv_a.loc[sample, "Split"]
            if "Split" in csv_a.columns
            else csv_b.loc[sample, "Split"],
        }
    )

# Step 6: Create a new DataFrame and save to CSV
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv(args.output, index=False)

print(f"New CSV file '{args.output}' has been created.")

"""python merge_by_prob.py 
--csv_a /path/to/output_predictions_pah.csv 
--csv_b /path/to/output_predictions_asd.csv --output /path/to/combined_predictions.csv"""

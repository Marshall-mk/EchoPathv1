import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse


# Function to load video and compute frame differences
def analyze_video(video_path, output_folder, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    # Extract frames from the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_count >= max_frames:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        frame_count += 1

    cap.release()

    # Compute frame differences
    diff_frames = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        diff_frames.append(diff)

    # Generate and save the first difference frame plot
    # plt.imshow(diff_frames[0], cmap="gray")
    # plt.title(f"Frame Difference - Frame 1 ({os.path.basename(video_path)})")
    # plt.colorbar()
    # diff_frame_plot_path = os.path.join(
    #     output_folder, f"{os.path.basename(video_path)}_diff_frame_1.png"
    # )
    # plt.savefig(diff_frame_plot_path)
    # plt.close()

    # Quantify frame differences using Mean Squared Difference (MSD)
    msd = [np.mean(df**2) for df in diff_frames]

    # Save MSD plot with enhanced styling
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    plt.plot(msd, linewidth=2, color="#A23B72", alpha=0.8)
    # plt.fill_between(range(len(msd)), msd, alpha=0.3, color='#A23B72') #2E86AB
    plt.title(
        f"Mean Squared Difference - {os.path.basename(video_path)}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Frames", fontsize=12, fontweight="semibold")
    plt.ylabel("Mean Squared Difference", fontsize=12, fontweight="semibold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    msd_plot_path = os.path.join(
        output_folder, f"{os.path.basename(video_path)}_msd_plot.png"
    )
    plt.savefig(msd_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save MSD data to a CSV file
    msd_df = pd.DataFrame({"Frame": range(1, len(msd) + 1), "MSD": msd})
    msd_csv_path = os.path.join(
        output_folder, f"{os.path.basename(video_path)}_msd.csv"
    )
    msd_df.to_csv(msd_csv_path, index=False)

    print(f"Results for {os.path.basename(video_path)} saved in {output_folder}")


# Function to process all videos in a folder
def analyze_folder(folder_path, output_folder, max_frames=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [
        f for f in os.listdir(folder_path) if f.endswith((".mp4", ".avi", ".mov"))
    ]

    # Loop through each video and analyze it
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Analyzing video: {video_file}")
        analyze_video(video_path, output_folder, max_frames)


# Main function to parse arguments and run the program
def main():
    parser = argparse.ArgumentParser(
        description="Analyze Echo Videos for Frame Differences"
    )
    parser.add_argument(
        "--input_folder", type=str, help="Path to the folder containing videos"
    )
    parser.add_argument(
        "--output_folder", type=str, help="Path to the folder to save results"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="Maximum number of frames to process per video (default: process all frames)",
    )

    args = parser.parse_args()

    # Analyze the folder
    analyze_folder(args.input_folder, args.output_folder, args.max_frames)


if __name__ == "__main__":
    main()

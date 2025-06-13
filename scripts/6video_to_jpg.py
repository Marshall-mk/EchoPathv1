import cv2
import os
import argparse


def extract_frames_from_videos(video_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the video folder
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)

        # Skip if it's not a video file
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        # Create a folder for the current video's frames
        video_name = os.path.splitext(video_file)[0]
        frames_folder = os.path.join(output_folder, video_name)
        os.makedirs(frames_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame as a JPG file
            frame_filename = os.path.join(frames_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_file} into {frames_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and save them as JPG images."
    )
    parser.add_argument(
        "--video_folder", type=str, help="Path to the folder containing video files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where frames will be saved.",
    )

    args = parser.parse_args()
    extract_frames_from_videos(args.video_folder, args.output_folder)

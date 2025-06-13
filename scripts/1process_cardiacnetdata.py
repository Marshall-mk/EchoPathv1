import os
import argparse
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
from tqdm import tqdm

"""

python scripts/1process_cardiacnetdatav1.py --folder_path /nfs/usrhome/khmuhammad/CardiacNet/ --output_format mp4 --fps 32 --resize 112,112

"""


def create_video_from_nifti(
    nifti_path, output_path, fps=32, normalize=True, resize=None
):
    """
    Convert a NIfTI volume to MP4 video

    Parameters:
    -----------
    nifti_path : str
        Path to the NIfTI file
    output_path : str
        Path to save the output video
    fps : int
        Frames per second for the output video
    normalize : bool
        Whether to normalize the pixel values to 0-255 range
    resize : tuple or None
        Target dimensions (width, height) for resizing frames, None for no resizing

    Returns:
    --------
    tuple:
        (success, width, height, num_frames) - Boolean success flag, frame dimensions, and frame count
    """
    try:
        # Load NIfTI file
        nifti_img = nib.load(nifti_path)
        volume_data = nifti_img.get_fdata()

        # rotate the volume data 90 degrees anti-clockwise
        volume_data = np.rot90(volume_data, k=1, axes=(0, 1))
        # flip the volume data along the y-axis
        volume_data = np.flip(volume_data, axis=1)

        # Determine the appropriate dimension for frames
        # Assuming the time dimension is the 3rd dimension (index 2)
        if len(volume_data.shape) >= 3:
            num_frames = volume_data.shape[2]
        else:
            raise ValueError(f"NIfTI volume {nifti_path} has less than 3 dimensions")

        # Get image dimensions
        height, width = volume_data.shape[0], volume_data.shape[1]

        # Set target dimensions for resizing
        if resize is not None:
            target_width, target_height = resize
        else:
            target_width, target_height = width, height

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        video = cv2.VideoWriter(
            output_path, fourcc, fps, (target_width, target_height), isColor=False
        )

        # Process each frame
        for i in range(num_frames):
            # Extract the frame
            frame = volume_data[:, :, i]

            # Normalize to 0-255 if requested
            if normalize:
                if np.max(frame) - np.min(frame) > 0:
                    frame = (
                        255 * (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
                    )
                else:
                    frame = np.zeros_like(frame)

            # Convert to uint8
            frame = frame.astype(np.uint8)

            # Resize if needed
            if resize is not None:
                frame = cv2.resize(frame, (target_width, target_height))

            # Write frame to video
            video.write(frame)

        # Release the video writer
        video.release()
        return True, target_width, target_height, num_frames

    except Exception as e:
        print(f"Error processing {nifti_path}: {e}")
        return False, 0, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI volumes to MP4/AVI videos"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="/nfs/usrhome/khmuhammad/CardiacNet/",
        help="Path to the root folder (e.g., /nfs/usrhome/khmuhammad/CardiacNet/)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="mp4",
        choices=["mp4", "avi"],
        help="Output video format (mp4 or avi)",
    )
    parser.add_argument(
        "--fps", type=int, default=32, help="Frames per second for the output videos"
    )
    parser.add_argument(
        "--resize",
        type=str,
        help="Resize output videos to WIDTH,HEIGHT (e.g., 256,256)",
    )

    args = parser.parse_args()

    # Parse resize parameter if provided
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split(","))
            resize = (width, height)
        except:
            print("Invalid resize format. Using original dimensions.")

    # Create output directory for videos
    videos_dir = os.path.join(args.folder_path, "Videos")
    os.makedirs(videos_dir, exist_ok=True)

    # Map classes to integers
    class_mapping = {}
    class_counter = 0

    # Data for CSV file
    csv_data = []

    # Walk through the directory
    print("Processing NIfTI files...")

    # Find all condition directories (CardiacNet-ASD, CardiacNet-PAH, etc.)
    condition_dirs = [
        d
        for d in os.listdir(args.folder_path)
        if os.path.isdir(os.path.join(args.folder_path, d))
        and d.startswith("CardiacNet-")
    ]

    for condition_dir in condition_dirs:
        condition_path = os.path.join(args.folder_path, condition_dir)
        condition_name = condition_dir.split("-")[1]  # Extract ASD, PAH, etc.

        # Find class directories (ASD, Non-ASD, PAH, Non-PAH)
        for class_dir in os.listdir(condition_path):
            class_path = os.path.join(condition_path, class_dir)

            if os.path.isdir(class_path) and class_dir not in ["Videos"]:
                # Form the full class name (e.g., ASD, Non-ASD)
                class_name = f"{class_dir}"

                # Add class to mapping if not already there
                if class_name not in class_mapping:
                    class_mapping[class_name] = class_counter
                    class_counter += 1

                # Process all subdirectories that contain NIfTI files
                for subdir in os.listdir(class_path):
                    subdir_path = os.path.join(class_path, subdir)

                    if os.path.isdir(subdir_path):
                        # Find patient NIfTI files in this directory
                        for file in os.listdir(subdir_path):
                            if True:
                                nifti_path = os.path.join(subdir_path, file)

                                original_name = subdir.replace("_image.nii", "")
                                patient_number = int(original_name)

                                # if class_dir.startswith("Non-"):
                                #     suffix = '_' + class_dir.split("-")[1]
                                # else:
                                #     suffix = ''
                                # Create patient_id with class prefix
                                patient_id = f"{class_name.lower()}_{patient_number}"

                                # Create output video filename that includes the condition
                                video_filename = f"{class_name.lower()}_{original_name}.{args.output_format}"
                                video_path = os.path.join(videos_dir, video_filename)
                                # remove file extension from video_filename for CSV
                                video_filename_no_ext = os.path.splitext(
                                    video_filename
                                )[0]

                                print(f"Converting {nifti_path} to {video_path}...")
                                success, frame_width, frame_height, num_frames = (
                                    create_video_from_nifti(
                                        nifti_path,
                                        video_path,
                                        fps=args.fps,
                                        resize=resize,
                                    )
                                )

                                if success:
                                    # Add entry to CSV data
                                    csv_data.append(
                                        {
                                            "video_path": video_path,
                                            "FileName": video_filename_no_ext,
                                            "condition": condition_name,
                                            "class_name": class_name,
                                            # "full_class": class_name,
                                            "class_id": class_mapping[class_name],
                                            "patient_id": patient_id,
                                            "FPS": args.fps,
                                            "FrameWidth": frame_width,
                                            "FrameHeight": frame_height,
                                            "NumberOfFrames": num_frames,
                                        }
                                    )

    # Create CSV file
    csv_path = os.path.join(args.folder_path, "FileList.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)

    print(f"\nProcessing complete!")
    print(f"Videos saved to: {videos_dir}")
    print(f"CSV mapping saved to: {csv_path}")
    print(f"Class mapping: {class_mapping}")
    print(f"Total videos processed: {len(csv_data)}")


if __name__ == "__main__":
    main()

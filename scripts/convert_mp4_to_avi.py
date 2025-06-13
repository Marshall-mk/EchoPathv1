import os
import sys
import argparse
import subprocess
from tqdm import tqdm
import logging
import glob
import shutil
import importlib.util


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("convert_mp4_to_avi.log"),
        ],
    )
    return logging.getLogger("mp4_to_avi")


def is_installed(package):
    """Check if a Python package is installed."""
    return importlib.util.find_spec(package) is not None


def find_ffmpeg():
    """Find the ffmpeg executable in common locations."""
    # Check if ffmpeg is in PATH
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")

    # Check common locations for ffmpeg
    common_locations = [
        # Linux/Mac locations
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        # Anaconda locations
        os.path.join(sys.prefix, "bin", "ffmpeg"),
        # Check for imageio's ffmpeg
        os.path.join(
            sys.prefix,
            "lib",
            "python" + sys.version[:3],
            "site-packages",
            "imageio_ffmpeg",
            "binaries",
        ),
    ]

    for location in common_locations:
        if os.path.isfile(location):
            return location
        elif os.path.isdir(location):
            # Search for ffmpeg binary in this directory
            for root, dirs, files in os.walk(location):
                for file in files:
                    if file.startswith("ffmpeg"):
                        return os.path.join(root, file)

    return None


def convert_with_ffmpeg(input_file, output_file, ffmpeg_path, logger):
    """Convert MP4 to AVI using FFmpeg with appropriate settings."""
    try:
        cmd = [
            ffmpeg_path,
            "-i",
            input_file,
            "-c:v",
            "rawvideo",  # Use raw video codec (uncompressed)
            "-pix_fmt",
            "rgb24",  # Specify pixel format
            "-y",  # Overwrite output file without asking
            output_file,
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Error converting {input_file} to {output_file}")
            logger.error(f"FFmpeg stderr: {stderr.decode('utf-8')}")
            return False
        return True
    except Exception as e:
        logger.error(f"Exception while converting {input_file}: {e}")
        return False


def convert_with_moviepy(input_file, output_file, logger):
    """Convert MP4 to AVI using moviepy."""
    try:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(input_file)
        clip.write_videofile(output_file, codec="rawvideo", fps=clip.fps)
        clip.close()
        return True
    except Exception as e:
        logger.error(f"Exception while converting with moviepy {input_file}: {e}")
        return False


def convert_with_imageio(input_file, output_file, logger):
    """Convert MP4 to AVI using imageio."""
    try:
        import imageio

        reader = imageio.get_reader(input_file)
        fps = reader.get_meta_data()["fps"]
        writer = imageio.get_writer(output_file, fps=fps)
        for frame in reader:
            writer.append_data(frame)
        writer.close()
        reader.close()
        return True
    except Exception as e:
        logger.error(f"Exception while converting with imageio {input_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 files to AVI format")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing MP4 files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save AVI files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="File pattern to match (default: *.mp4)",
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all MP4 files in input directory
    mp4_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    logger.info(f"Found {len(mp4_files)} MP4 files to convert")

    if len(mp4_files) == 0:
        logger.error(
            f"No MP4 files found in {args.input_dir} matching pattern {args.pattern}"
        )
        return

    # Try to find ffmpeg
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
        conversion_method = "ffmpeg"
    elif is_installed("moviepy"):
        logger.info("Using moviepy for conversion")
        conversion_method = "moviepy"
    elif is_installed("imageio"):
        logger.info("Using imageio for conversion")
        conversion_method = "imageio"
    else:
        logger.error(
            "Neither FFmpeg, moviepy, nor imageio is available. Cannot convert videos."
        )
        logger.error("Please install FFmpeg or run: pip install moviepy imageio")
        return

    # Convert each file
    success_count = 0
    for mp4_file in tqdm(mp4_files, desc="Converting files"):
        base_name = os.path.basename(mp4_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(args.output_dir, f"{name_without_ext}.avi")

        success = False
        if conversion_method == "ffmpeg":
            success = convert_with_ffmpeg(mp4_file, output_file, ffmpeg_path, logger)
        elif conversion_method == "moviepy":
            success = convert_with_moviepy(mp4_file, output_file, logger)
        else:  # imageio
            success = convert_with_imageio(mp4_file, output_file, logger)

        if success:
            success_count += 1

    logger.info(
        f"Conversion complete: {success_count}/{len(mp4_files)} files successfully converted"
    )


if __name__ == "__main__":
    main()

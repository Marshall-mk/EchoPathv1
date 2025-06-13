import os
import shutil

# List of source folders
source_folders = [
    "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv1",
    "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv2",
    # "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddpm_csv3",
    # "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddpm_csv4"
]
target_folder = (
    "/nfs/usrhome/khmuhammad/Echonet/samples/lvdm_dynamic_reproduced_with_ddim_csv"
)

# Create target subfolders
os.makedirs(os.path.join(target_folder, "jpg"), exist_ok=True)
os.makedirs(os.path.join(target_folder, "mp4"), exist_ok=True)

# Step 1: Find the number of samples in each source folder
num_samples_list = []
for source_folder in source_folders:
    jpg_path = os.path.join(source_folder, "jpg")
    subfolders = [
        f
        for f in os.listdir(jpg_path)
        if os.path.isdir(os.path.join(jpg_path, f)) and f.startswith("sample_")
    ]
    if subfolders:
        xxx_list = [int(f.split("_")[1]) for f in subfolders]
        max_xxx = max(xxx_list)
        num_samples = max_xxx + 1
    else:
        num_samples = 0
    num_samples_list.append(num_samples)

# Step 2: Compute cumulative offsets
offsets = [0]
for i in range(1, len(source_folders)):
    offsets.append(offsets[-1] + num_samples_list[i - 1])

# Step 3: Rename and move contents
for idx, source_folder in enumerate(source_folders):
    offset = offsets[idx]

    # Handle "jpg" subfolders
    jpg_path = os.path.join(source_folder, "jpg")
    for subfolder in os.listdir(jpg_path):
        full_path = os.path.join(jpg_path, subfolder)
        if os.path.isdir(full_path) and subfolder.startswith("sample_"):
            xxx = int(subfolder.split("_")[1])
            new_xxx = xxx + offset
            new_subfolder_name = f"sample_{new_xxx:06d}"
            shutil.move(
                full_path, os.path.join(target_folder, "jpg", new_subfolder_name)
            )

    # Handle "mp4" video files
    mp4_path = os.path.join(source_folder, "mp4")
    for file in os.listdir(mp4_path):
        full_path = os.path.join(mp4_path, file)
        if (
            os.path.isfile(full_path)
            and file.endswith(".mp4")
            and file.startswith("sample_")
        ):
            xxx = int(file.split("_")[1].split(".")[0])
            new_xxx = xxx + offset
            new_file_name = f"sample_{new_xxx:06d}.mp4"
            shutil.move(full_path, os.path.join(target_folder, "mp4", new_file_name))

print(f"Merged folders into {target_folder} with sequential renaming.")

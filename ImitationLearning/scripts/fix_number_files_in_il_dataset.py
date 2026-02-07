import os
import shutil

# Base paths for dataset and target folder for excessive files
base_path = os.path.join(os.path.dirname(__file__), "..", "..", "components", "data", "il_dataset")
too_many_path = os.path.join(os.path.dirname(__file__), "..", "data", "il_dataset_too_many")

# Create target folder if it does not exist
os.makedirs(too_many_path, exist_ok=True)

# Iterate over all subfolders (e.g., FloorPlan1, FloorPlan2, ...)
for scene_dir in os.listdir(base_path):
    scene_path = os.path.join(base_path, scene_dir)
    if not os.path.isdir(scene_path):
        continue  # Skip files

    # Get all files in the folder (any file type)
    files = [f for f in os.listdir(scene_path) if os.path.isfile(os.path.join(scene_path, f))]

    if len(files) > 10:
        # Create corresponding destination folder
        dest_dir = os.path.join(too_many_path, scene_dir)
        os.makedirs(dest_dir, exist_ok=True)
        # Select the files that are too many (sorted alphabetically for determinism)
        too_many_files = sorted(files)[10:]
        for fname in too_many_files:
            src = os.path.join(scene_path, fname)
            dst = os.path.join(dest_dir, fname)
            shutil.move(src, dst)
        print(f"{scene_dir}: {len(too_many_files)} files moved to {dest_dir}")

    elif len(files) < 10:
        print(f"{scene_dir} is missing {10 - len(files)} files")

    # Do nothing if exactly 10 files are present

print("Done.")

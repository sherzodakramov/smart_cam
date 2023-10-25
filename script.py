import os
import shutil

# Source directory containing all the images
source_dir = "clients"

# Destination directory where you want to move images
destination_dir = "train"

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        file_path = os.path.splitext(file)[0]
        os.makedirs(os.path.join(destination_dir, file_path), exist_ok=True)
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file_path)
        shutil.copy(source_path, destination_path)

print("Images moved to their respective directories.")

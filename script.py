import os
import shutil

# Source directory containing all the images
source_dir = "employees"

# Destination directory where you want to move images
destination_dir = "employees"

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(source_dir):
    for dir in dirs:
        for rt, ds, fs in os.walk(os.path.join(source_dir, dir)):
            for k in fs:
                source_path = os.path.join(source_dir, dir, k)
                shutil.move(source_path, destination_dir)
        os.removedirs(os.path.join(source_dir, dir))
        # file_path = os.path.splitext(file)[0]
        # os.makedirs(os.path.join(destination_dir, file_path), exist_ok=True)
        # destination_path = os.path.join(destination_dir, file_path)
        # shutil.copy(source_path, destination_path)

print("Images moved to their respective directories.")

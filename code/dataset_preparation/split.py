import os
import random
import shutil

source_folder = 'dataset/labels'  # e.g., folder containing label .txt files
output_folder_80 = 'dataset/labels/train'
output_folder_20 = 'dataset/labels/val'
os.makedirs(output_folder_80, exist_ok=True)
os.makedirs(output_folder_20, exist_ok=True)
split_ratio = 0.8

# Only get .txt files
all_files = [f for f in os.listdir(source_folder) if f.endswith('.txt') and os.path.isfile(os.path.join(source_folder, f))]
random.shuffle(all_files)

split_index = int(len(all_files) * split_ratio)
files_80 = all_files[:split_index]
files_20 = all_files[split_index:]

os.makedirs(output_folder_80, exist_ok=True)
os.makedirs(output_folder_20, exist_ok=True)

for file in files_80:
    shutil.copy(os.path.join(source_folder, file), os.path.join(output_folder_80, file))

for file in files_20:
    shutil.copy(os.path.join(source_folder, file), os.path.join(output_folder_20, file))

print(f"Copied {len(files_80)} files to {output_folder_80}")
print(f"Copied {len(files_20)} files to {output_folder_20}")

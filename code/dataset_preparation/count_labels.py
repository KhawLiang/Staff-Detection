import os

labels_path = r"staff_annotations"
dataset_train_path = r"dataset\labels\train"
dataset_val_path = r"dataset\labels\val"

# Get base filenames from each folder
all_labels = set(os.listdir(labels_path))
train_files = os.listdir(dataset_train_path)
val_files = os.listdir(dataset_val_path)

# Count how many files exist in labels_path for train and val
train_exist_count = sum(1 for f in train_files if f in all_labels)
val_exist_count = sum(1 for f in val_files if f in all_labels)

print(f"Total train files: {len(train_files)}")
print(f"Train files that exist in labels_path: {train_exist_count}")
print(f"Total val files: {len(val_files)}")
print(f"Val files that exist in labels_path: {val_exist_count}")
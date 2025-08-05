# A Python Script to get frame image by index in ./labels
import os
import shutil

def main():
    labels_path = r"dataset\labels\train"
    imgs_path = r"img"
    output_path = r"dataset\images\train"

    os.makedirs(output_path, exist_ok=True)

    for label_file in os.listdir(labels_path):
        if label_file.endswith(".txt"):
            # Get the frame index from the label filename
            frame_name = os.path.splitext(label_file)[0]
            img_filename = f"{frame_name}.jpg"
            img_path = os.path.join(imgs_path, img_filename)

            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(output_path, img_filename))
            else:
                print(f"Image not found: {img_path}")

if __name__ == "__main__":
    main()

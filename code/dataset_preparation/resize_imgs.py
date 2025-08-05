import cv2
import os

def resize_images_in_place(folder, size=(640, 640)):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            path = os.path.join(folder, filename)

            # Read the image
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to load image: {path}")
                continue

            # Resize
            resized_img = cv2.resize(img, size)

            # Overwrite the original image
            cv2.imwrite(path, resized_img)
            print(f"Resized: {path}")

if __name__ == "__main__":
    folder = r"dataset\imgs\val"
    resize_images_in_place(folder)

import os
import cv2

# -- Setup -- #
video_path= r"sample.mp4"
output_folder = r"img"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_num = 0

# -- Frame Extraction -- #
while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_num % 5 == 0:
        frame_path = os.path.join(output_folder, f"frame_{frame_num}.jpg")
        cv2.imwrite(frame_path, frame)
    frame_num += 1

cap.release()
import cv2
import datetime 
from ultralytics import YOLO

MODEL_PATH = r'staff_detection\train2\weights\best.pt'

VIDEO_PATH = r'sample.mp4'
# Get the current date and time and format it into a string
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create the output filename using the timestamp
OUTPUT_PATH = f'output_{current_time}.mp4'

def detect_and_show_video(model_path, video_path):
    """
    Loads a YOLO model, reads a video, and performs real-time staff detection
    with bounding boxes displayed on the video frames.
    """
    print("Initializing...")
    try:
        # Load the fine-tuned YOLOv8 model.
        model = YOLO(model_path)
        print(f"Model loaded successfully from '{model_path}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        # Open the video file.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: '{video_path}'")
        
        print(f"Video loaded successfully from '{video_path}'")
    except IOError as e:
        print(f"Error: {e}")
        return
    
    # Get video properties (width, height, fps) from the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Video will be saved to '{OUTPUT_PATH}'")
        
    print("Starting object detection. Press 'q' to exit.")

    # Loop through the video frames.
    while cap.isOpened():
        # Read a frame from the video.
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame.
            results = model(frame, stream=True, verbose=False)
            
            annotated_frame = frame.copy()

            # Process the results.
            for r in results:
                # Check if any objects were detected
                if r.boxes and len(r.boxes) > 0:
                    # Find the index of the bounding box with the highest confidence score.
                    # r.boxes.conf is a tensor of confidence scores for all detected boxes.
                    #print(f"r.boxes.conf: {r.boxes.conf}")
                    best_box_index = r.boxes.conf.argmax()
                    
                    # Get the single best bounding box object
                    best_box = r.boxes[best_box_index]
                    #print(f"bestbox: {best_box}")
                    #print(f"\nxyxy: {best_box.xyxy}")
                    
                    # Get the coordinates of the bounding box (xyxy format)
                    # We move the tensor to CPU, convert to numpy, and then to integer
                    xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    # Get the confidence score and class ID
                    conf = best_box.conf[0].cpu().numpy()
                    cls_id = int(best_box.cls[0].cpu().numpy())
                    
                    # Get the class name from the model's names list
                    class_name = r.names[cls_id]
                    
                    # Create the label text
                    label = f'{class_name} {conf:.2f}'
                    
                    # Draw the bounding box rectangle on the frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw the label background and text
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
            # Write the annotated frame to the output video file
            out.write(annotated_frame)

            # Display the annotated frame in a window.
            cv2.imshow("YOLOv8 Object Detection", annotated_frame)

            # Break the loop if the 'q' key is pressed.
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            # Break the loop if the end of the video is reached.
            break

    # Release the video capture object and close all display windows.
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Video saved to '{OUTPUT_PATH}'. Exiting.")


if __name__ == "__main__":
    detect_and_show_video(MODEL_PATH, VIDEO_PATH)
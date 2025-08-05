from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"original_model_weights\yolov8n.pt")

    model.train(data=r'C:\Users\khaw\2025\FootfallCam Test\dataset\dataset.yaml',       # Path to YAML config
                project='staff_detection',
                epochs=50,                          # Number of epochs
                imgsz=640,                          # Image size
                batch=8,                            # Batch size
                device=0)                           # GPU device index

    results = model.val()
    print(results)
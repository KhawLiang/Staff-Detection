import cv2
import sys
import datetime 
from ultralytics import YOLO

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage

MODEL_PATH = r'staff_detection\train2\weights\best.pt'

# --- DARK MODE STYLESHEET ---
DARK_MODE_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #2e2e2e; /* Dark background for window and central widget */
        color: #f0f0f0;            /* Light text color */
    }

    QLabel#video_label { /* Specific ID for the video label */
        background-color: black;
        color: white;
        font-family: "Segoe UI", sans-serif;
    }

    QPushButton {
        background-color: #555555; /* Darker gray for buttons */
        color: #f0f0f0;
        border: 1px solid #777777; /* Slightly lighter border */
        padding: 8px 15px;
        border-radius: 4px; /* Slightly rounded corners */
        font-family: "Segoe UI", sans-serif;
        font-size: 14px;
    }

    QPushButton:hover {
        background-color: #6a6a6a; /* Lighter gray on hover */
        border: 1px solid #888888;
    }

    QPushButton:pressed {
        background-color: #444444; /* Even darker when pressed */
    }

    QPushButton:disabled {
        background-color: #3a3a3a; /* Disabled state */
        color: #999999;
        border: 1px solid #5a5a5a;
    }

    QStatusBar {
        background-color: #3c3c3c; /* Darker status bar */
        color: #ffffff;
        border-top: 1px solid #4a4a4a; /* Separator line */
    }
"""
# --- END DARK MODE STYLESHEET ---

class MainWindow(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Staff detection")
        self.resize(1000, 500)

        # Set up main window layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.main_layout = QHBoxLayout(self.centralWidget)

        # Video display area
        self.video_label = QLabel("Please load a video file.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setFixedSize(640, 480)
        self.main_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Load model
        try:
            # Load the fine-tuned YOLOv8 model.
            # Make sure you have 'ultralytics' installed: pip install ultralytics
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from '{model_path}'")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Load video button
        # Select a video with staff in file explorer
        self.mp4_path = None
        self.select_video_btn = QPushButton("Load Video", self)
        self.select_video_btn.clicked.connect(self.load_video)

        # Start detection button
        self.start_detect_btn = QPushButton("Start Detection")
        self.start_detect_btn.clicked.connect(self.start_detection)
        self.start_detect_btn.setEnabled(False)

        # Stop detection button
        self.stop_detect_btn = QPushButton("Stop Detection")
        self.stop_detect_btn.clicked.connect(self.stop_detection)
        self.stop_detect_btn.setEnabled(False)

        # Layout for buttons
        buttonLayout = QVBoxLayout()

        buttonLayout.addStretch()

        # Set a fixed spacing between the buttons
        buttonLayout.setSpacing(10)

        buttonLayout.addWidget(self.select_video_btn)
        buttonLayout.addWidget(self.start_detect_btn)
        buttonLayout.addWidget(self.stop_detect_btn)

        buttonLayout.addStretch()

        self.main_layout.addLayout(buttonLayout)

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready. Load a video to begin.")

        # Set timer for frame processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .mp4 file",
            "",                                     # Start directory, empty string means current directory or last used
            "MP4 Files (*.mp4);;All Files (*.*)"    # Filter string
        )
        if file_path:
            self.mp4_path = file_path
            self.status_bar.showMessage(f"Video file selected: {self.mp4_path}")
            self.start_detect_btn.setEnabled(True)

            # Display the first frame as a preview
            cap = cv2.VideoCapture(self.mp4_path)
            if cap.isOpened():
                success, frame = cap.read()
                if success:
                    q_image = self.cv2_to_qimage(frame)
                    pixmap = QPixmap.fromImage(q_image)
                    self.video_label.setPixmap(pixmap.scaled(self.video_label.size(),
                                                             Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation))
                cap.release()

    def start_detection(self):
        if self.mp4_path and self.model:
            self.start_detect_btn.setEnabled(False)
            self.stop_detect_btn.setEnabled(True)
            self.status_bar.showMessage("Detection started...")

            # Open video capture
            self.video_capture = cv2.VideoCapture(self.mp4_path)
            if not self.video_capture.isOpened():
                self.status_bar.showMessage("Error: Cannot open video file.")
                self.stop_detection()
                return
            
            # Set up video writer for the output file
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f'output_{current_time}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
            
            # Start the timer, which will call process_frame() repeatedly
            self.timer.start(int(1000 / fps)) # Interval in milliseconds
    
    def process_frame(self):
        """
        Reads a single frame, performs detection, and updates the UI.
        This is called by the QTimer.
        """
        if self.video_capture and self.video_capture.isOpened():
            success, frame = self.video_capture.read()
            if success:
                results = self.model(frame, stream=True, verbose=False)
                annotated_frame = frame.copy()

                for r in results:
                    if r.boxes and len(r.boxes) > 0:
                        # get best conf box index
                        best_box_index = r.boxes.conf.argmax()

                        # get best box info
                        best_box = r.boxes[best_box_index]

                        xyxy = best_box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy

                        conf = best_box.conf[0].cpu().numpy()

                        cls_id = int(best_box.cls[0].cpu().numpy())
                        class_name = r.names[cls_id]
            
                        label = f'{class_name} {conf:.2f}'
                        coords_label = f'{class_name}: {x1}, {y1}, {x2}, {y2}'

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
                        cv2.putText(annotated_frame, coords_label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 

                        
                self.video_writer.write(annotated_frame)
                q_image = self.cv2_to_qimage(annotated_frame)
                self.update_frame(q_image)
            else:
                self.stop_detection()
                self.status_bar.showMessage(f"Video processing complete. Output video saved successfully to {self.output_path}")
        else:
            self.stop_detection()

    def stop_detection(self):
        self.timer.stop()
        self.start_detect_btn.setEnabled(True)
        self.status_bar.showMessage("Detection stopped.")
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.stop_detect_btn.setEnabled(False)

    def cv2_to_qimage(self, frame):
        '''
        Convert a cv2 load BGR frame to a PyQt QImage.
        '''
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return q_image
    
    def update_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply the dark mode stylesheet to the entire application
    app.setStyleSheet(DARK_MODE_STYLESHEET)
    window = MainWindow(MODEL_PATH)
    window.show()
    sys.exit(app.exec())
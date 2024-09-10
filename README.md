 Real-Time-Object-Detection-and-Depth-Estimation-Using-YOLO-and-MiDaS
Description:  This project integrates the YOLOv10 model for real-time object detection and the MiDaS model for depth estimation. Using live video feed, the system detects people and vehicles, calculates their depth, and makes decisions (e.g., “STOP” or “GO”) based on the proximity of detected objects. 

 README: Real-Time Object Detection and Depth Estimation using YOLOv10 and MiDaS

Overview
This project is a real-time object detection and depth estimation system built using the YOLOv10 model for detecting objects and the MiDaS model for estimating depth. The system works on live video feed, detects objects like people and vehicles, calculates their depth, and provides real-time decisions based on object proximity. CUDA is used to accelerate both object detection and depth estimation by utilizing GPU resources.

 How It Works
1. YOLOv10 for Object Detection:
   - The YOLOv10 model is responsible for detecting objects in the video frame, specifically focusing on "person" and "vehicle" classes.
   - The system checks if any objects in these categories are detected.
   
2. MiDaS for Depth Estimation:
   - Once an object is detected, the MiDaS model calculates the depth of the detected objects in the frame.
   - The depth map is normalized, and the brightest pixel within the bounding box is used to measure proximity.
   
3. Final Decision:
   - Based on the depth estimation, the system makes a real-time decision: 
     - If the brightest value exceeds the set threshold, the system outputs "STOP".
     - Otherwise, it outputs "GO".
   - The frame also displays the decision along with the FPS (frames per second) for performance analysis.

 Setup Instructions

 Prerequisites
- Python 3.6+
- A CUDA-enabled GPU and CUDA drivers (if using GPU acceleration).
- OpenCV (for video processing).
- PyTorch (for deep learning models).
- `ultralytics` package for YOLOv5/YOLOv10 models.

 Required Packages
Install the necessary Python packages:
```bash
pip install torch torchvision torchaudio
pip install opencv-python opencv-python-headless
pip install ultralytics
```

### Loading Models
This project uses two models:
- YOLOv10 for object detection (loaded via `ultralytics`).
- MiDaS for depth estimation (loaded via PyTorch Hub).

Both models are loaded at runtime.

Code Explanation
- YOLO Model Setup: The code loads the YOLOv10 model using the `ultralytics` library with a pretrained `yolov5l.pt` model. The model's class names are stored in `class_names` to detect "person" and various vehicle types.
  
- **MiDaS Model Setup**: MiDaS is used for depth estimation and is loaded via `torch.hub`. The input frame is transformed and passed to the model to generate a depth map.
  
- Video Feed: The project captures frames from a live webcam or a network camera feed using OpenCV's `VideoCapture`.
  
- Threading: The project uses threading to capture frames in parallel, ensuring smooth and real-time processing.
  
- Decision Logic: The system determines the proximity of objects using a threshold on the brightest pixel in the depth map.

How to Run
1. Connect to your webcam or provide an RTSP stream link.
2. Run the Python script:
   ```bash
   python main.py
   ```
3. The output window will display the live video feed with the following information:
   - Detected objects (e.g., people or vehicles).
   - Depth estimation using a colorized depth map.
   - The system's decision: "STOP" if an object is too close, "GO" otherwise.
   - Current FPS (Frames Per Second).

Expected Output
- The live video feed is shown with bounding boxes around detected people or vehicles.
- A depth map is displayed in the same window.
- If an object is too close (based on depth brightness), the system will display a red "STOP" message. Otherwise, it will display a green "GO" message.
- FPS is shown at the top of the window, indicating real-time performance.

Using CUDA
This project is designed to work with CUDA for faster performance on systems with NVIDIA GPUs. 
- **CUDA Setup**: Ensure that you have the necessary CUDA drivers installed on your machine.
- **CUDA Usage**: By default, the code will check if a GPU is available using:
  ```python
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  ```
  If CUDA is available, both the YOLO and MiDaS models will utilize the GPU for acceleration. If not, the system will run on the CPU, but it may not be as fast.

Key Features
- **Real-time Object Detection**: Uses YOLOv10 to detect objects like people and vehicles.
- **Real-time Depth Estimation**: Uses MiDaS to calculate depth and proximity.
- **Threading**: Frame capturing is done on a separate thread to improve processing speed.
- **CUDA Acceleration**: Leverages CUDA if available, ensuring the system performs at high FPS.

Troubleshooting
- **Webcam Issues**: If the webcam doesn't open, make sure the correct video feed URL or device is being used.
- **Model Download Issues**: If PyTorch Hub fails to download the MiDaS model, ensure that you have internet connectivity.
- **Performance**: If FPS is low, make sure that CUDA is enabled. Without CUDA, the process may be slower, especially with higher resolution inputs.

Dependencies
- `torch`: Provides deep learning support for running the models.
- `opencv-python`: Handles video capturing and frame processing.
- `ultralytics`: Used for YOLOv5/YOLOv10 object detection.
- `torch.hub`: Loads the MiDaS model for depth estimation.

Acknowledgements
- **YOLOv5/YOLOv10**: From [Ultralytics](https://github.com/ultralytics/yolov5)
- **MiDaS**: From [intel-isl](https://github.com/intel-isl/MiDaS)

Feel free to contribute and create pull requests for improvements!

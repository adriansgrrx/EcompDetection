# Electronics Component Detection

This project focuses on detecting various electronic components using a machine learning model (YOLOv8) and integrating the detection system with Arduino for additional functionalities. Below is the description of the key files and their purposes:

## Project Structure

### Dataset
- **Electronics Component Detection.v1i.yolov8**
  - Dataset for training the YOLOv8 model, generated using Roboflow.

### Arduino Sketch
- **ecomp_ard**
  - Arduino sketch for handling hardware functionalities related to the project.

### Python Programs
1. **best-wo-ard.py**
   - Main Python program for component detection without serial communication with Arduino.

2. **best-wo-tk.py**
   - Main Python program with serial communication to Arduino but without a graphical user interface (GUI). This is the latest and advisable version for deployment.

3. **best.py**
   - Main Python program with both serial communication and GUI.
   - Note: This version is outdated and may experience delays during camera initialization due to its complexity.

4. **cv-test.py**
   - Script for testing computer vision functionalities.

5. **serial-test.py**
   - Script for testing serial communication with Arduino.

6. **robo-test.py**
   - Script for testing integration with robotic components.

7. **onnx-test.py**
   - Script for testing the ONNX model's performance and inference.

8. **yolo-inference.py**
   - General-purpose script for performing inference using YOLO models.

### Machine Learning Models
- **YOLOv8s.pt**
  - YOLOv8 small model variant.

- **YOLOv8sv6.pt**
  - Version 6 of the YOLOv8 small model.

- **best.pt**
  - YOLOv8 model trained on the dataset for detecting electronic components. This file is outdated.

- **best.onnx**
  - ONNX version of the trained YOLOv8 model for compatibility with different platforms.

- **thbest.onnx**
  - Another ONNX version with optimized performance.

- **yolov5su.pt**
  - YOLOv5 model variant.

## Usage Instructions

### 1. Dataset Preparation
- The dataset is preprocessed and formatted using Roboflow. Ensure the dataset is properly set up before training or running detection programs.

### 2. Model Deployment
- Use `best-wo-tk.py` as the recommended Python program for deployment due to its stability and compatibility.
- Alternatively, choose the Python program based on the required functionality:
  - Use `best-wo-ard.py` if Arduino communication is not required.
  - Use `best.py` for the full feature set with GUI (not recommended due to its outdated status).
- Deploy models as needed:
  - Use `YOLOv8sv6.pt` or `YOLOv8s.pt` for YOLOv8 models.
  - Use `best.onnx` or `thbest.onnx` for ONNX-compatible platforms.

### 3. Testing
- Use the following scripts for testing and validation:
  - `cv-test.py` for computer vision-related operations.
  - `serial-test.py` for validating serial communication between the computer and Arduino.
  - `robo-test.py` for robotic component testing.
  - `onnx-test.py` for testing ONNX models.
  - `yolo-inference.py` for general-purpose model inference.

### 4. Arduino Integration
- Upload the `ecomp_ard` sketch to the Arduino board to enable hardware-related functionalities.

## Notes
- Ensure all dependencies (e.g., Python libraries, Arduino drivers) are installed before running the programs.
- If the camera initialization takes too long with `best.py`, consider using `best-wo-tk.py` for smoother performance.
- Outdated files, such as `best.py` and `best.pt`, should be avoided for deployment.

### Notes on Deployment Performance
- Deploying the YOLOv8 model on a Raspberry Pi 4 Model B (8GB RAM) without an accelerator like Google Coral TPU results in significantly low FPS (0.31 FPS), making it unsuitable for live capture deployment.  
- In comparison, deploying the model on a laptop with an integrated GPU achieves a much higher FPS (20.78 FPS), ensuring smoother performance.

![FPS Comparison](https://drive.google.com/uc?id=1oC4vNBp3vY5fOwv3Aoy4c8Nm0U6EqxUc)

> *Figure: FPS comparison between Raspberry Pi and Laptop with integrated GPU.*

## Acknowledgments
- Dataset generated using Roboflow.
- Machine learning model trained using YOLOv8.

## Future Enhancements
- Optimize camera initialization for GUI-enabled programs.
- Expand the dataset to include more electronic components.
- Improve integration between software and hardware modules for enhanced performance.


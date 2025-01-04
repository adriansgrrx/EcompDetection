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
   - Main Python program with serial communication to Arduino but without a graphical user interface (GUI).

3. **best.py**
   - Main Python program with both serial communication and GUI.
   - Note: This version may experience delays during camera initialization due to its complexity.

4. **cv-test.py**
   - Script for testing computer vision functionalities.

5. **serial-test.py**
   - Script for testing serial communication with Arduino.

### Machine Learning Models
- **best.pt**
  - YOLOv8 model trained on the dataset for detecting electronic components.

- **thbest.onnx**
  - YOLOv8 model converted to ONNX format for compatibility with different platforms.

## Usage Instructions
1. **Dataset Preparation**
   - The dataset is preprocessed and formatted using Roboflow. Ensure the dataset is properly set up before training or running detection programs.

2. **Model Deployment**
   - Use `best.pt` or `thbest.onnx` depending on the platform compatibility.
   - Run the desired Python program for detection based on the required functionality:
     - Use `best-wo-ard.py` if Arduino communication is not required.
     - Use `best-wo-tk.py` for serial communication without a GUI.
     - Use `best.py` for the full feature set with GUI.

3. **Testing**
   - Use `cv-test.py` for testing computer vision-related operations.
   - Use `serial-test.py` for validating serial communication between the computer and Arduino.

4. **Arduino Integration**
   - Upload the `ecomp_ard` sketch to the Arduino board to enable hardware-related functionalities.

## Notes
- Ensure all dependencies (e.g., Python libraries, Arduino drivers) are installed before running the programs.
- If the camera initialization takes too long with `best.py`, consider testing with the other Python programs to isolate the issue.

## Acknowledgments
- Dataset generated using Roboflow.
- Machine learning model trained using YOLOv8.

## Future Enhancements
- Optimize camera initialization for the GUI-enabled program.
- Expand the dataset to include more electronic components.
- Improve integration between software and hardware modules for enhanced performance.

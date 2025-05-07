
# EcompDetection

**Automated Sorting and Defect Detection of Electronic Components Using Computer Vision for PUP CPE Laboratory**

EcompDetection is a computer vision-based system designed to assist in the inspection, classification, and sorting of electronic components using a YOLOv8 model. This project is developed for the PUP Computer Engineering Laboratory to aid in the educational and quality assurance processes.

## 🔍 Features

- **Object Detection:** Identifies various electronic components and defects using a custom-trained YOLOv8 model.
- **Edge TPU Optimization:** Includes TensorFlow Lite models compiled for Coral Edge TPU to accelerate inference.
- **GUI Application:** A full-screen Tkinter-based GUI for real-time detection, object counting, and serial communication with Arduino.
- **Automated Sorting:** Interfaces with a conveyor or robotic arm system for physical sorting based on detection results.
- **Logging and Analysis:** Captures detection results and logs them for further analysis.

## 📁 Project Structure

```
EcompDetection/
│
├── ai models/                    # Contains AI model training files and data
├── ecomp_ard/                    # Arduino-related code and hardware interface
├── py files/                     # Supporting Python scripts and modules
├── servo-test/                   # Tests for robotic arm servos
├── tflite new/                   # Updated TFLite models (EdgeTPU, INT8, etc.)
│
├── 240_ecomp_yolov8n.pt          # YOLOv8 PyTorch model
├── 240_yolov8n_edgetpu.tflite    # Edge TPU-compiled TFLite model
├── 240_yolov8n_int8.tflite       # INT8 quantized TFLite model
│
├── convert.py                    # Model conversion utilities
├── coral-ecomp-detect.py         # Inference using Coral Edge TPU
├── detect.py                     # Standard inference script
├── detect_copy.py                # Backup of detect.py
├── ecomp-detect-yolov8n_edgetpu.tflite  # Another Edge TPU model variant
│
├── ecomp-gui.py                  # Full-screen GUI for real-time detection
├── inference-gui.py              # Simplified GUI version
├── inference.py                  # Base inference script
├── inference_ard.py              # Inference script with Arduino communication
├── sort.py                       # Logic for sorting detected components
├── yolov8n.pt                    # Backup of base YOLOv8 model
```

## 🖥️ Requirements

- Python 3.9.12
- OpenCV 4.5.5.62
- PyCoral
- Tkinter
- Ultralytics 8.2.73
- Serial Communication (pyserial)
- Edge TPU Runtime (for Coral USB Accelerator)
- Arduino IDE (for firmware)

## 🚀 Running the Application

📌 **Note:** It is recommended to create and activate a virtual environment before running the application to avoid dependency conflicts.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Inference test:**
```bash
python inference.py
```

**With Arduino communication enabled:**
```bash
python inference_ard.py
```

## 🤖 Hardware Integration

- **Camera:** USB webcam (mounted behind or above robotic arm)
- **Processing:** Raspberry Pi 4B 8GB with Coral USB Accelerator
- **Actuator:** Servo Motor (controlled via Arduino UNO)
- **Components:** Resistors, capacitors, LEDs, and defective parts (e.g., rusted, cracked, missing leg)

## 📌 Status

- [/] Model trained and converted
- [/] Real-time GUI developed
- [/] Edge TPU tested successfully
- [/] Arduino integration done
- [ ] Final deployment and enclosure

## 📜

This project is developed for academic purposes at Polytechnic University of the Philippines and is open for educational and research use.

---

**Developed by BSCpE - Group4202**

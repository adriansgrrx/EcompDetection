import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import serial
import threading
import queue
from collections import defaultdict

# -------------------------------------
# Arduino Setup
# -------------------------------------
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Change COM port if needed
time.sleep(2)  # Allow Arduino time to reset

# Queue and cooldown management
component_queue = queue.Queue()
cooldown_tracker = defaultdict(lambda: 0)
cooldown_period = 5  # seconds

# Active object tracker
active_objects = defaultdict(list)  # class -> list of (cx, cy)

# Arduino communication thread
def arduino_communication():
    while True:
        try:
            component = component_queue.get(timeout=1)
            if component:
                print(f"Sending to Arduino: {component}")
                arduino.write(component.encode())

                # Wait for Arduino reply
                response_timeout = time.time() + 2
                while time.time() < response_timeout:
                    if arduino.in_waiting > 0:
                        response = arduino.readline().decode().strip()
                        print(f"Arduino response: {response}")
                        if response == "DONE":
                            break
                else:
                    print("Arduino response timeout.")
        except queue.Empty:
            time.sleep(0.5)

# Start Arduino communication
arduino_thread = threading.Thread(target=arduino_communication, daemon=True)
arduino_thread.start()

# -------------------------------------
# YOLOv8 Detection + Counting + Arduino
# -------------------------------------
original_classes = ['BJT', 'LED', 'burnt', 'capacitor', 'cracked', 'faded', 'missing-leg', 'resistor', 'rust']
defect_classes = {'burnt', 'cracked', 'faded', 'missing-leg', 'rust'}
count_classes = ['BJT', 'LED', 'capacitor', 'resistor', 'defective', 'unknown']  # Added 'unknown'

model = YOLO('ecomp-detect-yolov8n_edgetpu.tflite', task='detect')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

# Frame dimensions
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define central rectangle area (centered box)
rect_w, rect_h = 400, 480  # size of the trigger box
rect_x1 = (frame_w - rect_w) // 2
rect_y1 = (frame_h - rect_h) // 2
rect_x2 = rect_x1 + rect_w
rect_y2 = rect_y1 + rect_h

print("Live inference with counting started. Press 'q' to quit.")

min_distance = 30
centroid_timeout = 5.0

# Add tracking for unknown components
no_detection_timer = 0
no_detection_threshold = 5.0  # seconds to wait before considering "unknown"
last_detection_time = time.time()
unknown_component_sent = False

counts = {cls: 0 for cls in count_classes}
seen_centroids = {cls: [] for cls in count_classes}

def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, imgsz=256, conf=0.7, verbose=False)
    detections = result[0].boxes
    annotated_frame = result[0].plot()

    current_time = time.time()
    
    # Check for any detections in the rectangle
    found_detection_in_rect = False

    for cls in count_classes:
        seen_centroids[cls] = [(cx, cy, t) for cx, cy, t in seen_centroids[cls] if current_time - t < centroid_timeout]

    # Cleanup active_objects after timeout
    for cls in list(active_objects.keys()):
        active_objects[cls] = [(cx, cy, t) for cx, cy, t in active_objects[cls] if current_time - t < centroid_timeout]

    if detections is not None and detections.xyxy is not None and len(detections.xyxy) > 0:
        for i, box in enumerate(detections.xyxy):
            cls_id = int(detections.cls[i].item())
            original_class = original_classes[cls_id]
            class_name = 'defective' if original_class in defect_classes else original_class

            if class_name not in counts:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = get_centroid((x1, y1, x2, y2))
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)

            if rect_x1 <= cx <= rect_x2 and rect_y1 <= cy <= rect_y2:
                found_detection_in_rect = True
                last_detection_time = current_time
                unknown_component_sent = False  # Reset flag when a detection is found
                
                already_tracked = False
                for prev_cx, prev_cy, _ in active_objects[class_name]:
                    distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    if distance < min_distance:
                        already_tracked = True
                        break

                if not already_tracked:
                    counts[class_name] += 1
                    seen_centroids[class_name].append((cx, cy, current_time))
                    active_objects[class_name].append((cx, cy, current_time))
                    print(f"Detected {class_name} at {datetime.now().strftime('%H:%M:%S')} - Count: {counts[class_name]}")

                    if current_time - cooldown_tracker[class_name] > cooldown_period:
                        cooldown_tracker[class_name] = current_time
                        if class_name == 'BJT':
                            component_queue.put('A')
                        elif class_name == 'LED':
                            component_queue.put('B')
                        elif class_name == 'capacitor':
                            component_queue.put('C')
                        elif class_name == 'defective':
                            component_queue.put('D')
                        elif class_name == 'resistor':
                            component_queue.put('E')
                        # else:
                        #     component_queue.put('F')
    
    # Handle unknown component detection
    elapsed_time = current_time - last_detection_time
    
    # Display time since last detection (for debugging)
    cv2.putText(annotated_frame, f'Time since last detection: {elapsed_time:.1f}s', 
                (10, frame_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Check if we need to handle an unknown component
    if elapsed_time >= no_detection_threshold and not unknown_component_sent:
        # Check if there's something in the rectangle that's not recognized
        # We need to analyze the frame to see if there's any object in the rectangle
        # This is a simple check using contours and motion detection
        
        # Extract the rectangle region
        roi = frame[rect_y1:rect_y2, rect_x1:rect_x2]
        
        # Convert to grayscale and apply blur to reduce noise
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Use a simple threshold to find contours
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if there are any significant contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
        
        if significant_contours:
            print(f"Unknown component detected at {datetime.now().strftime('%H:%M:%S')}")
            counts['unknown'] += 1
            component_queue.put('U')  # Send 'U' for unknown component
            unknown_component_sent = True
            last_detection_time = current_time  # Reset timer
    
    # Draw central trigger rectangle
    cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)

    # Display counts
    y_offset = 30
    for i, (cls, count) in enumerate(counts.items()):
        cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display FPS
    fps = 1.0 / (time.time() - tic)
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (annotated_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Edge TPU Detection + Counting', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
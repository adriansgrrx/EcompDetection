import cv2
from ultralytics import YOLO
import serial
import time
import threading
import queue
from collections import defaultdict

# Initialize serial communication with Arduino
arduino = serial.Serial('COM6', 9600, timeout=1)  # Replace 'COM6' with your Arduino's port
time.sleep(2)  # Allow time for the connection to establish

# Load the YOLOv8 model
model = YOLO("best.pt")

# Shared queue for FIFO
component_queue = queue.Queue()

# Cooldown dictionary to track the last detection time for each component type
cooldown_tracker = defaultdict(lambda: 0)
cooldown_period = 2  # Seconds to wait before adding the same component again

# Arduino communication thread
def arduino_communication():
    while True:
        if not component_queue.empty():
            print(f"Queue before processing: {list(component_queue.queue)}")  # Debug: Print queue contents
            component = component_queue.get()  # Get the first component from the queue
            print(f"Processing component: {component}")
            
            arduino.write(component.encode())  # Send command to Arduino
            
            # Wait for Arduino confirmation
            while True:
                if arduino.in_waiting > 0:
                    response = arduino.readline().decode().strip()
                    print(f"Arduino response: {response}")  # Debug: Print Arduino response
                    if response == "DONE":
                        print(f"Component {component} processed successfully.")
                        break  # Move to the next component
        else:
            print("Queue is empty. Waiting for new components...")  # Debug: Empty queue state
            time.sleep(0.5)  # Wait briefly if the queue is empty

# Start Arduino communication thread
arduino_thread = threading.Thread(target=arduino_communication, daemon=True)
arduino_thread.start()

# Function to handle YOLO detection and video feed
def start_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Process detection results
        for detection in results[0].boxes:
            cls = int(detection.cls)
            confidence = detection.conf

            if confidence > 0.70:  # Process only high-confidence detections
                current_time = time.time()
                if current_time - cooldown_tracker[cls] > cooldown_period:
                    # Update the cooldown tracker for this component type
                    cooldown_tracker[cls] = current_time

                    # Add the detected component to the queue
                    if cls == 0:
                        component_queue.put('A')  # BJT
                        print("Status: BJT detected!")
                    elif cls == 1:
                        component_queue.put('B')  # LED
                        print("Status: LED detected!")
                    elif cls == 2:
                        component_queue.put('C')  # Capacitor
                        print("Status: Capacitor detected!")
                    elif cls == 3:
                        component_queue.put('D')  # Defective component
                        print("Status: Defective component detected!")
                    elif cls == 4:
                        component_queue.put('E')  # Resistor
                        print("Status: Resistor detected!")
                    else:
                        component_queue.put('F')  # Unknown
                        print("Status: Unknown component detected!")

                    print(f"Queue after adding: {list(component_queue.queue)}")  # Debug: Print queue after addition

        # Display the annotated frame in a window
        annotated_frame = results[0].plot()
        cv2.imshow("Ecomp Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the detection
            break

    cap.release()
    cv2.destroyAllWindows()

# Start detection in a separate thread
def run_detection():
    threading.Thread(target=start_detection).start()

# Start detection when the script is run
if __name__ == "__main__":
    run_detection()

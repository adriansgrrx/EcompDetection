from ultralytics import YOLO
import cv2
import serial
import time
import threading

# Initialize serial communication with Arduino
arduino = serial.Serial('COM6', 9600, timeout=1)  # Replace 'COM6' with your Arduino's port
time.sleep(2)  # Allow time for the connection to establish

# Load the YOLOv8 model
model = YOLO("best.pt")

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
            cls = int(detection.cls)  # Get the class index
            confidence = detection.conf  # Get confidence score

            if confidence > 0.70:  # Process only high-confidence detections
                if cls == 0:  # BJT detected
                    arduino.write(b'A')
                    print("Status: BJT detected!")
                elif cls == 1:  # LED detected
                    arduino.write(b'B')
                    print("Status: LED detected!")
                elif cls == 2:  # Capacitor detected
                    arduino.write(b'C')
                    print("Status: Capacitor detected!")
                elif cls == 3:  # Defective component detected
                    arduino.write(b'D')
                    print("Status: Defective component detected!")
                elif cls == 4:  # Resistor detected
                    arduino.write(b'E')
                    print("Status: Resistor detected!")
                else:
                    arduino.write(b'F')  # Unknown
                    print("Status: Unknown component detected!")
            else:  # Low-confidence detections
                arduino.write(b'F')  # Unknown
                print("Status: Unknown component detected!")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the detection
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()

# Start detection in a separate thread
def run_detection():
    threading.Thread(target=start_detection).start()

# Start detection when the script is run
if __name__ == "__main__":
    run_detection()

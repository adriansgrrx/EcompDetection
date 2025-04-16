import time
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('ecompy_yolov8_edgetpu.tflite', task='detect')

# Define your custom class names
class_names = ['BJT', 'LED', 'burnt', 'capacitor', 'cracked',
               'defective', 'faded', 'missing-leg', 'resistor', 'rust']

# Start video capture
cap = cv2.VideoCapture(0)

print('Live inference started. Press "q" to exit.')
while cap.isOpened():
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    output = model(frame, imgsz=320, verbose=False)

    # Process detections
    for det in output[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if score < 0.6:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = class_names[int(class_id)] if int(class_id) < len(class_names) else f'class_{int(class_id)}'

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    # Add FPS display
    fps = int(1 / (time.time() - tic))
    cv2.putText(frame, f'FPS: {fps}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Live Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the webcam or video capture (0 = default webcam, you can change it to 1, 2, etc., for other cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the live stream.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with bounding boxes, labels, etc.

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

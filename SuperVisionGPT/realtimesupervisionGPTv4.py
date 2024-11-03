import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Define traffic-related labels
TRAFFIC_OBJECTS = ['traffic light', 'stop sign', 'parking meter']

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Processing frames in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run YOLO model inference on the frame (disable printing stats)
    results = model(frame, verbose=False)[0]  # Set verbose=False to suppress console output

    # Flag to check if traffic-related object is detected
    traffic_detected = False

    # Iterate over detections
    for result in results.boxes:
        class_id = int(result.cls)  # Get the class ID
        label = model.names[class_id]  # Get the label for the detected class

        # Check if the detected object is traffic-related
        if label in TRAFFIC_OBJECTS:
            print(f"ALERT: {label} detected!")
            traffic_detected = True
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for traffic objects
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            # If any non-traffic object is detected, print "obstacle ahead"
            print("Obstacle ahead!")
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for other obstacles

    # Display the frame with detections
    cv2.imshow('Traffic Sign Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

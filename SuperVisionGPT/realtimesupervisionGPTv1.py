import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (use the appropriate model trained for traffic signals detection)
model = YOLO('yolov8n.pt')  # or 'yolov8n.pt' for the nano version (faster but less accurate)

# Load traffic signal labels (you can fine-tune this with your dataset)
# Common traffic signal classes might include: 'traffic light'
TRAFFIC_SIGNAL_LABELS = ['traffic light']

# Start video capture (0 = default webcam, or replace with video path)
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

    # Run the YOLO model inference on the frame
    results = model(frame)[0]  # Get the detection results from the model

    # Iterate over detections and filter traffic signals
    for result in results.boxes:
        class_id = int(result.cls)  # Get the class ID
        label = model.names[class_id]  # Get the label for the detected class
        
        # Check if the detected object is a traffic signal
        if label in TRAFFIC_SIGNAL_LABELS:
            # Draw bounding box and label on the frame
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            confidence = result.conf.item()  # Confidence score

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Blue text

    # Display the frame with detections
    cv2.imshow('Traffic Signal Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

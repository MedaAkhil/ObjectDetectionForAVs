import cv2
import supervision as sv
from inference import get_model

# Load the YOLO model
model = get_model(model_id="yolov8n-640")

# Start video capture from the webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()  # Read a frame from the video stream
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Run the YOLO model on the current frame
    results = model.infer(frame)[0]

    # Convert the inference results to detections format
    detections = sv.Detections.from_inference(results)

    # Optionally: Draw detections on the frame (boxes, labels, etc.)
    annotated_frame = sv.plot_detections(frame, detections)

    # Display the annotated frame in a window
    cv2.imshow("Real-Time YOLOv8 Detection", annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

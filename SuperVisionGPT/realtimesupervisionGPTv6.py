import cv2
from ultralytics import YOLO
model = YOLO('yolov10n.pt')
INPUT_CAMERA_RECORDING = 0
TRAFFIC_OBJECTS = ['traffic light', 'stop sign', 'parking meter']
cap = cv2.VideoCapture(INPUT_CAMERA_RECORDING)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    results = model(frame, verbose=False)[0] 
    traffic_detected = False
    object_detected = False
    for result in results.boxes:
        object_detected = True
        class_id = int(result.cls)
        label = model.names[class_id]
        confidence = result.conf[0]
        if label in TRAFFIC_OBJECTS:
            print(f"ALERT: {label} detected with confidence {confidence:.2f}!")
            traffic_detected = True
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            print(f"Obstacle ahead! {label} detected with confidence {confidence:.2f}")
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    if not object_detected:
        print("Keep going")
        cv2.putText(frame, "Keep going", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow('Traffic Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
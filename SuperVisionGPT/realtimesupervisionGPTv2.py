import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

TRAFFIC_SIGN_LABELS = ['stop sign', 'traffic light', 'speed limit', 'yield sign', 'no entry', 'turn left', 'turn right']

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    results = model(frame)[0]

    for result in results.boxes:
        class_id = int(result.cls)
        label = model.names[class_id]
        
        if label in TRAFFIC_SIGN_LABELS:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf.item() 

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Traffic Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

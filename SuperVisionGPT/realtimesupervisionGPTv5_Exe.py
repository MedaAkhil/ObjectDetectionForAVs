import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
INPUT_CAMERA_RECORDING = 1
TRAFFIC_OBJECTS = ['traffic light', 'stop sign', 'parking meter']
traffic_signs = [
    "Stop",
    "Keep Left",
    "Keep Right",
    "No Entry",
    "One Way",
    "Two Way Traffic",
    "No Parking",
    "No Stopping",
    "Pedestrian Crossing",
    "School Zone",
    "Speed Limit 30",
    "Speed Limit 50",
    "Speed Limit 70",
    "Speed Limit 90",
    "Speed Limit 100",
    "Speed Limit 120",
    "End of Speed Limit",
    "U-Turn",
    "No U-Turn",
    "Right Turn Only",
    "Yield",
    "Roundabout",
    "Left Turn Only",
    "No Left Turn",
    "No Right Turn",
    "Merge",
    "Lane Ends",
    "Divided Highway",
    "End Divided Highway",
    "Traffic Signal Ahead",
    "Railroad Crossing",
    "Slippery Road",
    "Falling Rocks",
    "Steep Hill Down",
    "Steep Hill Up",
    "Narrow Bridge",
    "Low Clearance",
    "Sharp Turn Right",
    "Sharp Turn Left",
    "Winding Road",
    "Truck Route",
    "No Trucks",
    "Weight Limit",
    "Height Limit",
    "Width Limit",
    "Dead End",
    "No Through Road",
    "Road Work Ahead",
    "Detour",
    "Road Closed",
    "Cattle Crossing",
    "Speed Bump",
    "Hospital",
    "Emergency Services",
    "Bus Stop",
    "Motorway",
    "End Motorway",
    "Pedestrian Zone",
    "End Pedestrian Zone",
    "Parking Area",
    "Rest Area",
    "No Bicycles",
    "No Pedestrians",
    "No Motor Vehicles",
    "No Overtaking",
    "End No Overtaking",
    "No Horn",
    "End No Horn",
    "No Littering",
    "No Smoking",
    "Gas Station",
    "Electric Charging Station",
    "Hotel",
    "Restaurant",
    "Restroom",
    "Scenic View",
    "Information",
    "Police",
    "Fire Station",
    "Roundabout Ahead",
    "Crossroad",
    "T-Junction",
    "Y-Junction",
    "End of Lane",
    "Caution",
    "Tunnel Ahead",
    "Bridge Ahead",
    "Slow",
    "Give Way",
    "Stop Sign Ahead",
    "School Crossing"
]
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
import cv2

# Open a connection to the webcam (0 is typically the default camera)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was read correctly, ret will be True
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame in a window
    cv2.imshow('Webcam Video', frame)

    # Wait for 1 millisecond between frames and check if 'q' is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

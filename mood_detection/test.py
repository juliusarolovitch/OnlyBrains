import cv2

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video device.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    video_capture.release()
    cv2.destroyAllWindows()

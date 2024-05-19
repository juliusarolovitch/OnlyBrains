import cv2
from fer import FER

def detect_emotions_in_real_time():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = FER()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotions = detector.detect_emotions(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draws bounding boxes + detected emotions around found faces
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the detected emotions with their confidence scores
            y_offset = y - 10 if y - 10 > 10 else y + 10
            for idx, (emo, score) in enumerate(emotion['emotions'].items()):
                text = f"{emo}: {score:.2f}"
                cv2.putText(frame, text, (x, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions_in_real_time()

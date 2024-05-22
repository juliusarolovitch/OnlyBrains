import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Function for calculating MAR (here: vert/horiz)
def calculate_mar(landmarks):
    mouth_indices = [61, 291, 0, 17]  

    A = np.linalg.norm(np.array(landmarks[mouth_indices[0]]) - np.array(landmarks[mouth_indices[1]]))  # vertical distance
    B = np.linalg.norm(np.array(landmarks[mouth_indices[2]]) - np.array(landmarks[mouth_indices[3]]))  # horizontal distance
    mar = A / B
    return mar

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]

            mar = calculate_mar(landmarks)
            
            for idx in [61, 291, 0, 17]:
                cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (landmarks[0][0], landmarks[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            

            if mar < 2.3:  # Threshold value for MAR best in range 2.0 - 2.5
                cv2.putText(frame, "Speaking", (landmarks[0][0], landmarks[0][1]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

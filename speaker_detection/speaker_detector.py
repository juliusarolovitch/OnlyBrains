import cv2
import base64
from openai import OpenAI
import numpy as np
import mediapipe as mp

class SpeakerDetector:
    ## Captures one frame from webcam
    def frame_capture_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Unable to open webcam")

        ret, frame = cap.read()
        if not ret:
            raise ValueError("Unable to capture frame from webcam")

        brightness_factor = 4 
        brightened_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=50)
        cap.release()
        return brightened_frame

    ## Takes one frame from webcam and returns b64 encoding of frame 
    def webcam_frame_tob64(self):
        frame = self.frame_capture_webcam()
        cv2.imshow('Brightened Frame', frame)
        cv2.waitKey(0)  

        # Encode to JPEG (same as previous versions)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise ValueError("Unable to encode frame as JPEG")
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        cv2.destroyAllWindows()
        return jpg_as_text
    
    def calculate_MAR(self, landmarks): 
        mouth_indices = [61, 291, 0, 17]  # upper lip, lower lip, left corner, right corner

        A = np.linalg.norm(np.array(landmarks[mouth_indices[0]]) - np.array(landmarks[mouth_indices[1]])) 
        B = np.linalg.norm(np.array(landmarks[mouth_indices[2]]) - np.array(landmarks[mouth_indices[3]])) 
        mar = A / B
        return mar
    
    ##Captures one frame from webcam and identifies whether the person within it is speaking, can vary number of people detected using num_faces parameter
    def MAR_frame(self, num_faces): 
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=num_faces, min_detection_confidence=0.5)

        def is_speaking_in_frame(frame):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]
                    mar = self.calculate_MAR(landmarks)
                    
                    for idx in [61, 291, 0, 17]:
                        cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (landmarks[0][0], landmarks[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    if mar < 2.3: # Threshold value for MAR best in range 2.0 - 2.5
                        cv2.putText(frame, "Speaking", (landmarks[0][0], landmarks[0][1]-30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        return frame, True
            
            return frame, False

        frame = self.frame_capture_webcam()

        processed_frame, is_speaking = is_speaking_in_frame(frame)
        print(f"Is speaking: {is_speaking}")
        
        # Display the frame
        cv2.imshow("Frame", processed_frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

        return is_speaking
    
    ## Captures a video feed from webcam, live annotates faces detected with whether the model deems the person in frame is speaking, can modify number of faces recognized by model using num_faces parameter.
    def MAR_video(self, num_faces): 
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=num_faces, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]

                    mar = self.calculate_MAR(landmarks)
                    
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

    ## Captures webcam frame and makes API call to gpt-4o with desired prompt and role as well as the image captured by the webcam. 
    def gpt4o_frame_webcam(self, API_key, prompt, role):
        client = OpenAI(api_key=API_key)
        frame = self.webcam_frame_tob64()

        response = client.chat.completions.create(
            model = "gpt-4o", 
            messages = [{"role": "system", "content": f"{role}"},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"{prompt}"}, 
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64, {frame}"}
                            }
                        ]}], 
                        temperature=0.0,
            )
    
        return response
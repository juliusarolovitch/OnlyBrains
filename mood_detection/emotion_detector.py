import cv2
from fer import FER
import base64
from openai import OpenAI

class EmotionDetector: 
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
    

    ## Captures a webcam frame, detects emotions for the people within the frame, returns a dominant emotion 
    def FER_frame_webcam(self):
        frame = self.frame_capture_webcam()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #### Detecting Emotions: 
        detector = FER()
        emotions = detector.detect_emotions(frame_rgb)

        if not emotions:
            print("No face detected")
            return

        dominant_emotion, emotion_score = detector.top_emotion(frame_rgb)

        ### Code for displaying the one captured frame corresponding to the emotion
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Captured Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return dominant_emotion
    
    ##
    def FER_video_webcam(self):
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


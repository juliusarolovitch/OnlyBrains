from emotion_detector import EmotionDetector
    
API_key = ""
prompt = "What are the dominant emotions expressed by the person in this image?"
role = "You are an emotion recognition assistant who detects a person's dominant emotions from a provided image. You are to provide a textual description of the persons emotions. "
detector = EmotionDetector()
print(detector.FER_frame_webcam())
detector.FER_video_webcam()
print((detector.gpt4o_frame_webcam(API_key, prompt, role)).choices[0].message.content)





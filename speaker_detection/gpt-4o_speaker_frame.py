import cv2
import base64
from openai import OpenAI
import time

### API Key config code 
MODEL="gpt-4o"
API_KEY = "sk-proj-9UJXW5YKGSuVH1NKsNlzT3BlbkFJW93qCLfqDk61s3XRT4ao"

client = OpenAI(api_key=API_KEY)

def capture_frame_and_encode():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError("Unable to open webcam")

    ret, frame = cap.read()

    if not ret:
        raise ValueError("Unable to capture frame from webcam")

    brightness_factor = 4 
    brightened_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)


    # cv2.imshow('Brightened Frame', brightened_frame)
    # cv2.waitKey(0)  
    cap.release()

    # Encode to JPEG (same as previous versions)
    ret, buffer = cv2.imencode('.jpg', brightened_frame)
    if not ret:
        raise ValueError("Unable to encode frame as JPEG")
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    cv2.destroyAllWindows()

    return jpg_as_text

### Testing code 
#capture_frame_and_encode()

### API Call Code:
start = time.time()

captured_frame_b64 = capture_frame_and_encode()

response = client.chat.completions.create(
    model = MODEL, 
    messages = [{"role": "system", "content": "You are a video speech detection assistent, your job is to scan an image and determine whether a person in the image is speaking and who the main speaker in the image is"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Who is the main speaker in the image, if anyone is speaking?"}, 
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64, {captured_frame_b64}"}
                    }
                ]}], 
                temperature=0.0,
)

print(response.choices[0].message.content)
end = time.time()
print(end-start)


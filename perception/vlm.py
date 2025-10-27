import os
import queue
import sounddevice as sd
import cv2
import torch
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from PIL import Image

model_path = "vosk-model-small-en-us-0.15"
speech_model = Model(model_path)
recognizer = KaldiRecognizer(speech_model, 16000)
q = queue.Queue()

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)

def capture_frame():
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Failed to open camera")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture frame")
        return None

    frame = cv2.flip(frame, -1)
    frame = cv2.resize(frame, (320, 240)) # need to resize since too heavy 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = Image.fromarray(frame)
    return frame

def speech_to_text():
    print("Recording in 3 seconds...")
    time.sleep(3)
    
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, channels=1, dtype='int16', callback=lambda indata, frames, time, status: q.put(bytes(indata))):
            print("recording...)
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    print("Speech Recognized:", result)
                    return result
    except Exception as e:
        print(f"Error during recording: {e}")

def run_moondream(image, user_prompt):
	if image is None:
		print("No image captured.")
		return
		
	encoded_image = model.encode_image(image)
	response = model.query(encoded_image, user_prompt)["answer"]

	print("VLM Response:")
	print(response)

if __name__ == "__main__":
    print("capturing image")
    image = capture_frame()

    user_prompt = speech_to_text()

    if user_prompt:
        run_moondream(image, user_prompt)

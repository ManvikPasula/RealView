import os
import queue
import sounddevice as sd
import cv2
import torch
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from gtts import gTTS
import pygame
import time
import numpy as np
import json

print("Loading Vosk Model...")
model_path = "vosk-model-small-en-us-0.15"
speech_model = Model(model_path)
recognizer = KaldiRecognizer(speech_model, 16000)
q = queue.Queue()
print("Vosk loaded")

print("Loading Moondream2 Vision model...")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)
print("Moondream2 Model Loaded")

pygame.mixer.init()

def beep(times=1):
    for _ in range(times):
        os.system('play -nq -t alsa synth 0.1 sine 300 vol 0.1')  
        time.sleep(0.2)

def play_audio(text):
    try:
        print("Generating response audio...")
        tts = gTTS(text=text, lang="en")
        audio_file = "/tmp/response.mp3"
        tts.save(audio_file)
        
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        os.remove(audio_file)
        print("Response spoken")

    except Exception as e:
        print(f"Error during audio playback: {e}")

def display_camera():
    cap = cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Failed to open camera")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame = cv2.flip(frame, -1) 
        frame = cv2.resize(frame, (320, 240))
        og_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Camera Feed", og_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            print("Capturing image from camera...")

            captured_frame = Image.fromarray(frame)
            if captured_frame is not None:
                print("Image Captured Successfully!")

                beep(times=1)

                user_prompt = speech_to_text()
                if user_prompt:
                    run_moondream(captured_frame, user_prompt)

        elif key == ord('q'):
            print("Exiting program...")
            beep(times=1)
            break

    cap.release()
    cv2.destroyAllWindows()

def speech_to_text():
    #print("Recording in 3 seconds...")
    #time.sleep(3)
    
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, channels=1, dtype='int16', callback=lambda indata, frames, time, status: q.put(bytes(indata))):
            print("Recording...")
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
    
    try:
        print("Encoding image...")
        encoded_image = model.encode_image(image)
        print("Fixing Prompt...")
        parsed = json.loads(user_prompt)
        prompt = parsed["text"]
        print("Running Vision-Language Model...")
        response = model.query(encoded_image, prompt)["answer"]

        print("Moondream2 Vision Response:")
        print(response)

        play_audio(response)

    except Exception as e:
        print(f"Error during vision-language model processing: {e}")

def main():
    print("System is starting...")

    beep(times=3)

    print("'p' to capture and process, 'q' to exit.")
    display_camera()

if __name__ == "__main__":
    main()

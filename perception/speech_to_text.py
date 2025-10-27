import os
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

model_path = "vosk-model-small-en-us-0.15"
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=16000, blocksize=8000, channels=1, dtype='int16', callback=callback):
    print("Listening...")
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            print(recognizer.Result())

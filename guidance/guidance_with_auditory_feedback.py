from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import time
import skimage.io
import cv2
from models import __models__
from utils import *
import gc
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import sounddevice as sd
import threading

# cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = "cuda" if torch.cuda.is_available() else "cpu"
side_model = torch.load('segformer_model.pth', map_location=device, weights_only=False)
side_model.eval()
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

camera_id1 = "/dev/video2"  # Left camera

FOV_HORIZONTAL = 140
FRAME_CENTER_X = 640 // 2

BUZZ_FREQ = 440  # Frequency in Hz
SAMPLE_RATE = 44100
VOLUME_SCALE = 0.2  

left_volume = 0.0
right_volume = 0.0
running = True
sidewalk_detected = False

t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
wave = np.sin(2 * np.pi * BUZZ_FREQ * t).astype(np.float32)

def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    outdata[:, 0] = wave[:frames] * left_volume
    outdata[:, 1] = wave[:frames] * right_volume

def start_audio_feedback():
    with sd.OutputStream(channels=2, samplerate=SAMPLE_RATE, callback=audio_callback):
        while running:
            sd.sleep(50)

audio_thread = threading.Thread(target=start_audio_feedback, daemon=True)
audio_thread.start()

def update_buzz(offset_deg):
    global left_volume, right_volume, sidewalk_detected
    
    max_offset = FOV_HORIZONTAL / 2
    
    if offset_deg is not None:
        sidewalk_detected = True
        volume = min(abs(offset_deg) / max_offset, 1.0) * VOLUME_SCALE
        
        if offset_deg > 5:
            left_volume = 0.0
            right_volume = volume
        elif offset_deg < -5:
            left_volume = volume
            right_volume = 0.0
        else:
            left_volume = 0.0
            right_volume = 0.0
    else:
        sidewalk_detected = False
        left_volume = 0.0
        right_volume = 0.0

def play_startup_beeps():
    beep_freq = 1000  # Hz
    beep_duration = 0.1  # seconds
    pause_duration = 0.2  # seconds
    volume = 0.5

    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration), endpoint=False)
    beep = np.sin(2 * np.pi * beep_freq * t).astype(np.float32)

    for _ in range(3):
        sd.play(np.column_stack([beep * volume, beep * volume]), samplerate=SAMPLE_RATE)
        sd.wait()
        time.sleep(pause_duration)

def calculate_offset(mask):
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        pixel_offset = cX - FRAME_CENTER_X
        offset_deg = (pixel_offset * FOV_HORIZONTAL) / 640
        return offset_deg
    return None

def test():
    play_startup_beeps()

    cap_left = cv2.VideoCapture(camera_id1, cv2.CAP_V4L2)
    
    frame_count = 0

    while cap_left.isOpened():
        succes_left, frame_left = cap_left.read()
        
        if not succes_left:
            print("Failed to capture frames")
            break

        frame_left = cv2.flip(frame_left, -1)
        frame = frame_left.copy()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = side_model(**inputs).logits
            pred_mask = torch.nn.functional.interpolate(outputs, size=(480, 640), mode="bilinear", align_corners=False)
            pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()

            mask = (pred_mask > 0.5).astype(np.uint8) * 255
            overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            offset_deg = calculate_offset(mask)
            update_buzz(offset_deg)

            if sidewalk_detected:
                direction = f"Move {offset_deg:.1f} degrees {'RIGHT' if offset_deg > 0 else 'LEFT'}"
                cv2.putText(combined, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("SegFormer Live Inference", combined)
            frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cv2.destroyAllWindows()

# Stop audio on exit
def stop_audio():
    global running
    running = False
    audio_thread.join()

import atexit
atexit.register(stop_audio)

if __name__ == '__main__':
    print("Starting...")
    test()

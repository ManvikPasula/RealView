#!/usr/bin/env python3
from __future__ import print_function, division
import os
import queue
import threading
import time
import json
import atexit
import gc
import cv2
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import skimage.io
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForCausalLM, SegformerImageProcessor
from PIL import Image
from gtts import gTTS
import pygame
from models import __models__
from utils import *
 
 
exit_flag = False          
running = True            
vlm_processing = False    
depth_enabled = False    
guidance_enabled = False  
latest_vlm_frame = None    
 
high_disparity_mode = False
left_volume = 0.0
right_volume = 0.0
beep_index = 0
sidewalk_detected = False
 
try:
    current_output_device = sd.default.device[1]
    if current_output_device == -1:
        raise Exception("No valid output device set")
except Exception as e:
    print("No valid default output device found. Attempting to set one.")
    devices = sd.query_devices()
    current_output_device = None
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            sd.default.device = (None, i)
            current_output_device = i
            print(f"Set default output device to index {i}: {dev['name']}")
            break
if current_output_device is None:
    raise Exception("No valid audio output device found.")
 
 
print("Loading Vosk Model...")
vosk_model_path = "vosk-model-small-en-us-0.15"
speech_model = Model(vosk_model_path)
print("Vosk loaded")
 
print("Loading Moondream2 Vision model (VLM)...")
vlm_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)
print("Moondream2 VLM Loaded")
 
pygame.mixer.init()
print("Mixer initialized")
 
print("Loading Depth Model...")
maxdisp = 192
attention_weights_only = False
model_name = 'Fast_ACVNet_plus'
loadckpt = 'generalization.ckpt'
output_dir = './output'
 
depth_model = __models__[model_name](maxdisp, attention_weights_only)
depth_model = nn.DataParallel(depth_model)
depth_model.cuda()
 
print(f"Loading Depth model weights from {loadckpt}")
state_dict = torch.load(loadckpt)
depth_model.load_state_dict(state_dict['model'])
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading SegFormer segmentation model (Guidance)...")
side_model = torch.load('segformer_model.pth', map_location=device, weights_only=False)
side_model.eval()
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
 
params_to_use = 'stereoMap.xml'
cv_file = cv2.FileStorage()
cv_file.open(params_to_use, cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
 
camera_left_id = "/dev/video2"  
camera_right_id = "/dev/video0"    
 
os.makedirs(output_dir, exist_ok=True)
 
FOV_HORIZONTAL = 140
FRAME_CENTER_X = 640 // 2
 
BUZZ_FREQ = 440      
SAMPLE_RATE = 44100
VOLUME_SCALE = 0.2
 
_beep_duration = 1.0  # seconds
_num_samples = int(SAMPLE_RATE * _beep_duration)
_t_beep = np.linspace(0, _beep_duration, _num_samples, endpoint=False)
high_disparity_beep_buffer = 0.1 * np.sin(2 * np.pi * 400 * _t_beep).astype(np.float32)
 
t_cont = 0.5 * np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
wave = np.sin(2 * np.pi * BUZZ_FREQ * t_cont).astype(np.float32)
 
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
 
def speech_to_text_prompt():
    prompt_q = queue.Queue()
    prompt_recognizer = KaldiRecognizer(speech_model, 16000)
   
    def prompt_callback(indata, frames, time_info, status):
        prompt_q.put(bytes(indata))
   
    try:
        print("Recording prompt...")
        with sd.RawInputStream(samplerate=16000, blocksize=8000, channels=1,
                                 dtype='int16', callback=prompt_callback):
            while True:
                data = prompt_q.get()
                if prompt_recognizer.AcceptWaveform(data):
                    result = prompt_recognizer.Result()
                    print("Prompt recognized:", result)
                    return result
    except Exception as e:
        print(f"Error during prompt recording: {e}")
        return None
 
def run_vlm(image, user_prompt):
    if image is None:
        print("No image captured for VLM.")
        return
    try:
        image_np = np.array(image)
        cv2.imshow("VLM Image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
       
        print("Encoding image for VLM...")
        encoded_image = vlm_model.encode_image(image)
        print("Fixing Prompt...")
        parsed = json.loads(user_prompt)
        prompt = parsed["text"]
        print("Running Moondream2 VLM query...")
        response = vlm_model.query(encoded_image, prompt)["answer"]
 
        print("VLM Response:")
        print(response)
        play_audio(response)
       
        cv2.destroyWindow("VLM Image")
    except Exception as e:
        print(f"Error during VLM processing: {e}")
 
@make_nograd_func
def test_sample(left_img, right_img):
    depth_model.eval()
    disp_ests = depth_model(left_img, right_img)
    return disp_ests[-1]
 
def calculate_offset(mask):
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        pixel_offset = cX - FRAME_CENTER_X
        offset_deg = (pixel_offset * FOV_HORIZONTAL) / 640
        return offset_deg
    return None
 
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
    beep_freq = 1000   # Hz
    beep_duration = 0.1  # seconds
    pause_duration = 0.2  # seconds
    volume = 0.5
 
    t_beep = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration), endpoint=False)
    beep_sound = np.sin(2 * np.pi * beep_freq * t_beep).astype(np.float32)
 
    for _ in range(3):
        sd.play(np.column_stack([beep_sound * volume, beep_sound * volume]), samplerate=SAMPLE_RATE)
        sd.wait()
        time.sleep(pause_duration)
 
def audio_callback(outdata, frames, time_info, status):
    global beep_index, high_disparity_mode
    if status:
        print(status)
    if high_disparity_mode:
        buffer_len = len(high_disparity_beep_buffer)
        end_index = beep_index + frames
        if end_index <= buffer_len:
            samples = high_disparity_beep_buffer[beep_index:end_index]
            beep_index = end_index % buffer_len
        else:
            part1 = high_disparity_beep_buffer[beep_index:]
            part2 = high_disparity_beep_buffer[:(end_index - buffer_len)]
            samples = np.concatenate((part1, part2))
            beep_index = end_index - buffer_len
        outdata[:] = np.column_stack([samples, samples])
    else:
        outdata[:, 0] = wave[:frames] * left_volume
        outdata[:, 1] = wave[:frames] * right_volume
 
def start_audio_feedback():
    with sd.OutputStream(device=current_output_device,
                         channels=2, samplerate=SAMPLE_RATE,
                         callback=audio_callback):
        while running:
            sd.sleep(50)
 
def depth_guidance_loop():
    global exit_flag, latest_vlm_frame, high_disparity_mode
    cap_left = cv2.VideoCapture(camera_left_id, cv2.CAP_V4L2)
    cap_right = cv2.VideoCapture(camera_right_id, cv2.CAP_V4L2)
 
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Failed to open stereo cameras for depth/guidance.")
        return
 
    play_startup_beeps()
 
    while not exit_flag:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
       
        if not ret_left or not ret_right:
            print("Failed to capture frames from stereo cameras")
            break
 
        raw_left = cv2.flip(frame_left, -1)
        raw_right = cv2.flip(frame_right, -1)
        latest_vlm_frame = raw_left.copy()
       
        cv2.imshow("Left Camera", raw_left)
        cv2.imshow("Right Camera", raw_right)
 
        if depth_enabled or guidance_enabled:
            flipped_left = cv2.flip(frame_left, -1)
            flipped_right = cv2.flip(frame_right, -1)
 
            frame_rgb = cv2.cvtColor(flipped_left, cv2.COLOR_BGR2RGB)
 
            if depth_enabled:
                remapped_left = cv2.remap(flipped_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4)
                remapped_right = cv2.remap(flipped_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4)
               
                left_img = cv2.cvtColor(remapped_left, cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(remapped_right, cv2.COLOR_BGR2RGB)
                left_img = left_img.astype(np.float32) / 255.0
                right_img = right_img.astype(np.float32) / 255.0
 
                left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).cuda()
                right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).cuda()
           
                disp_est_np = tensor2numpy(test_sample(left_tensor, right_tensor))
                disp_est = np.array(disp_est_np[0], dtype=np.float32)
                disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
 
                roi = disp_est[120:480, 160:480]
                if np.any(roi >= 170):
                    high_disparity_mode = True
                else:
                    high_disparity_mode = False
 
                disp_est_norm = cv2.normalize(disp_est, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                disp_est_norm = np.uint8(disp_est_norm)
                disp_vis = cv2.applyColorMap(disp_est_norm, cv2.COLORMAP_JET)
 
                cv2.imshow('Disparity', disp_vis)
           
            if guidance_enabled:
                inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = side_model(**inputs).logits
                    pred_mask = torch.nn.functional.interpolate(outputs, size=(480, 640), mode="bilinear", align_corners=False)
                    pred_mask = torch.sigmoid(pred_mask).cpu().numpy().squeeze()
 
                    mask = (pred_mask > 0.5).astype(np.uint8) * 255
                    overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    combined = cv2.addWeighted(raw_left, 0.7, overlay, 0.3, 0)
 
                    offset_deg = calculate_offset(mask)
                    update_buzz(offset_deg)
 
                    if sidewalk_detected:
                        direction = f"Move {offset_deg:.1f} degrees {'RIGHT' if offset_deg > 0 else 'LEFT'}"
                        cv2.putText(combined, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
 
                    cv2.imshow("SegFormer Guidance", combined)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break
 
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print("Depth/guidance processing terminated.")
 
 
def keyboard_listener():
    global exit_flag, vlm_processing, depth_enabled, guidance_enabled
    cv2.namedWindow("Command Window")
    while not exit_flag:
        command_img = np.zeros((100,600,3), dtype=np.uint8)
        cv2.putText(command_img, "Keyboard: q=exit, p=VLM, d=depth on, f=depth off, g=guidance on, h=guidance off",
                    (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow("Command Window", command_img)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            exit_flag = True
            break
        elif key == ord('p'):
            if not vlm_processing:
                vlm_processing = True
                beep(times=1)
                if latest_vlm_frame is not None:
                    pil_image = Image.fromarray(cv2.cvtColor(latest_vlm_frame, cv2.COLOR_BGR2RGB))
                    prompt_result = speech_to_text_prompt()
                    if prompt_result:
                        run_vlm(pil_image, prompt_result)
                else:
                    print("No VLM frame available.")
                vlm_processing = False
        elif key == ord('d'):
            depth_enabled = True
            print("Depth enabled via keyboard.")
        elif key == ord('f'):
            depth_enabled = False
            print("Depth disabled via keyboard.")
        elif key == ord('g'):
            guidance_enabled = True
            print("Guidance enabled via keyboard.")
        elif key == ord('h'):
            guidance_enabled = False
            print("Guidance disabled via keyboard.")
    cv2.destroyWindow("Command Window")
 
def wake_word_listener():
    global exit_flag, vlm_processing, depth_enabled, guidance_enabled
    wake_q = queue.Queue()
    wake_recognizer = KaldiRecognizer(speech_model, 16000)
   
    def wake_callback(indata, frames, time_info, status):
        wake_q.put(bytes(indata))
   
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, channels=1,
                                 dtype='int16', callback=wake_callback):
            print("Wake word listener active. Awaiting commands...")
            while not exit_flag:
                data = wake_q.get()
                if wake_recognizer.AcceptWaveform(data):
                    result = wake_recognizer.Result()
                    try:
                        result_dict = json.loads(result)
                    except Exception as e:
                        print("Error parsing wake word result:", e)
                        continue
                    recognized_text = result_dict.get("text", "").strip().lower()
                    print("Wake word recognized:", recognized_text)
                    if "exit key" in recognized_text:
                        print("Exit command received via wake word.")
                        exit_flag = True
                        beep(times=3)
                        break
                    elif "real view activate" in recognized_text or "jarvis" in recognized_text:
                        if not vlm_processing:
                            vlm_processing = True
                            beep(times=1)
                            if latest_vlm_frame is not None:
                                pil_image = Image.fromarray(cv2.cvtColor(latest_vlm_frame, cv2.COLOR_BGR2RGB))
                                prompt_result = speech_to_text_prompt()
                                if prompt_result:
                                    run_vlm(pil_image, prompt_result)
                            else:
                                print("No VLM frame available.")
                            vlm_processing = False
                    elif "enable distance" in recognized_text:
                        depth_enabled = True
                        print("Depth enabled")
                        beep(times=1)
                    elif "disable distance" in recognized_text:
                        depth_enabled = False
                        print("Depth disabled")
                        beep(times=2)
                    elif "enable guidance" in recognized_text:
                        guidance_enabled = True
                        print("Guidance enabled")
                        beep(times=1)
                    elif "disable guidance" in recognized_text:
                        guidance_enabled = False
                        print("Guidance disabled")
                        beep(times=2)
                time.sleep(0.1)
    except Exception as e:
        print(f"Error in wake word listener: {e}")
   
    print("Wake word listener terminated.")
 
def stop_audio():
    global running
    running = False
 
atexit.register(stop_audio)
 
def main():
    global exit_flag
    print("System is starting...")
    beep(times=3)
   
    audio_thread = threading.Thread(target=start_audio_feedback, daemon=True)
    audio_thread.start()
   
    depth_thread = threading.Thread(target=depth_guidance_loop)
    depth_thread.start()
   
    wake_thread = threading.Thread(target=wake_word_listener)
    wake_thread.start()
   
    kb_thread = threading.Thread(target=keyboard_listener)
    kb_thread.start()
   
    wake_thread.join()
    kb_thread.join()
    exit_flag = True
    depth_thread.join()
   
    stop_audio()
    print("Exiting system...")
 
if __name__ == "__main__":
    main()

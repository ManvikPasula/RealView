import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def record_audio(filename, duration=5, samplerate=44100):
	print(f"recording for {duration} seconds")
	
	device_id = 0
	
	try:
		print("Recording...")
		audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
		sd.wait()
		
		print("saving to filename")
		write(filename, samplerate, audio)
		
		print("recording complete")
	except Exception as e:
		print(f"error: {e}")
		
if __name__ == "__main__":
	output_file = "output.wav"
	record_audio(output_file, duration=5)

from __future__ import print_function, division
import argparse
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

# cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACV)')
parser.add_argument('--model', default='Fast_ACVNet_plus', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--loadckpt', default='generalization.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--attention_weights_only', default=False, type=str, help='only train attention weights')
parser.add_argument('--output_dir', default='./output', help='directory to save output disparity maps')
args = parser.parse_args()

model = __models__[args.model](args.maxdisp, args.attention_weights_only)
model = nn.DataParallel(model)
model.cuda()

print(f"Loading model from {args.loadckpt}")
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

params_to_use = 'stereoMap.xml'
cv_file = cv2.FileStorage()
cv_file.open(params_to_use, cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

camera_id1 = "/dev/video2" 
camera_id2 = "/dev/video0"

os.makedirs(args.output_dir, exist_ok=True)

# Test function
@make_nograd_func
def test_sample(left_img, right_img):
	model.eval()
	disp_ests = model(left_img, right_img)
	return disp_ests[-1]

def test():
	cap_left = cv2.VideoCapture(camera_id1, cv2.CAP_V4L2)
	cap_right = cv2.VideoCapture(camera_id2, cv2.CAP_V4L2)
	
	frame_count = 0

	while cap_right.isOpened() and cap_left.isOpened():
		succes_right, frame_right = cap_right.read()
		succes_left, frame_left = cap_left.read()
		
		if not succes_left or not succes_right:
			print("Failed to capture frames")
			break

		frame_right = cv2.flip(frame_right, -1)
		frame_left = cv2.flip(frame_left, -1)

		frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
		frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

		frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
		frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
		
		frame_left = frame_left.astype(np.float32) / 255.0
		left_img = torch.from_numpy(frame_left).permute(2, 0, 1).unsqueeze(0).cuda()
		
		frame_right = frame_right.astype(np.float32) / 255.0
		right_img = torch.from_numpy(frame_right).permute(2, 0, 1).unsqueeze(0).cuda()

		start_time = time.time()
		disp_est_np = tensor2numpy(test_sample(left_img, right_img))
		print('Inference time = {:.3f}s'.format(time.time() - start_time))

		disp_est = np.array(disp_est_np[0], dtype=np.float32)
		disp_est_uint = np.round(disp_est * 256).astype(np.uint16)

		#disp_vis = cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.03), cv2.COLORMAP_JET)
		disp_est_norm = cv2.normalize(disp_est, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		disp_est_norm = np.uint8(disp_est_norm)  # Convert to 8-bit for color mapping
		disp_vis = cv2.applyColorMap(disp_est_norm, cv2.COLORMAP_JET)
		
		filename = os.path.join(args.output_dir, f'disp_latest_{frame_count}.png')
		print(f"Saving disparity map to {filename}")
		skimage.io.imsave(filename, disp_vis)

		cv2.imshow('Disparity', disp_vis)
		cv2.imshow('Left Image', frame_left)
		cv2.imshow('Right Image', frame_right)
		
		frame_count += 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap_left.release()
	cap_right.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	test()

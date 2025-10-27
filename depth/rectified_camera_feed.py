import numpy as np
import cv2

camera_id1 = "/dev/video2"  
camera_id2 = "/dev/video0"

params_to_use = 'stereoMap.xml'

cv_file = cv2.FileStorage()
cv_file.open(params_to_use, cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

cap_left = cv2.VideoCapture(camera_id1, cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(camera_id2, cv2.CAP_V4L2)

while (cap_right.isOpened() and cap_left.isOpened()):
	succes_right, frame_right = cap_right.read()
	succes_left, frame_left = cap_left.read()
	
	frame_right = cv2.flip(frame_right, -1)
	frame_left = cv2.flip(frame_left, -1)
        
	frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
	frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
	
	cv2.imshow("combined frames", cv2.hconcat([frame_left, frame_right]))
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
cap_left.release()
cap_right.release()

cv2.destroyAllWindows()

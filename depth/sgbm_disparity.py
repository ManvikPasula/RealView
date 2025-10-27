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

min_disp = 0
num_disp = 16 * 6  # Must be divisible by 16
block_size = 9

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,
    P2=32 * 3 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

cap_left = cv2.VideoCapture(camera_id1, cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(camera_id2, cv2.CAP_V4L2)

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

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    disparity_clean = cv2.medianBlur(disparity.astype(np.uint8), 5)

    disp_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    disp_map_color = cv2.applyColorMap(disp_map, cv2.COLORMAP_JET)

    combined_frames = cv2.hconcat([frame_left, frame_right])

    cv2.imshow("Combined Frames (Left, Right)", combined_frames)
    cv2.imshow("Disparity", disp_map_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

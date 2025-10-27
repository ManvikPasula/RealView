# undistort code for left camera, right camera code is exact same but "left" and "right" names are swapped

import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 

images = sorted(glob.glob('./images/left_image_*.png'))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        #cv.drawChessboardCorners(img, (9,6), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(100)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h=480
w=640
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

camera_id = "/dev/video2"

cap = cv.VideoCapture(camera_id, cv.CAP_V4L2)
if not cap.isOpened():
	print(f"Error: Could not open camera {camera_id}")
else:
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Error: Failed to capture frame")
			Break

		frame = cv.flip(frame, -1)

		dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
		 
		#x, y, w, h = roi
		#dst = dst[y:y+h, x:x+w]

		combined_frames = np.concatenate((frame, dst), axis=1)

		cv.imshow('Original (Left) vs Undistorted (Right)', combined_frames)

		if cv.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv.destroyAllWindows()


np.savez_compressed("./camera_params_left.npz", 
                    mtx=mtx, 
                    dist=dist, 
                    newcameramtx=newcameramtx)
print("\nCalibration data saved to 'camera_params_left.npz'")

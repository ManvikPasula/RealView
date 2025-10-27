import numpy as np
import cv2 as cv
import glob

chessboardSize = (9, 6)
frameSize = (640, 480)

criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 20

objpoints = []
imgpointsL = []
imgpointsR = []

#imagesLeft = sorted(glob.glob('./images/left_image_*.png'))
#imagesRight = sorted(glob.glob('./images/right_image_*.png'))
imagesLeft = sorted(glob.glob('./undistorted_images/undistorted_left_*.png'))
imagesRight = sorted(glob.glob('./undistorted_images/undistorted_right_*.png'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
	imgL = cv.imread(imgLeft)
	imgR = cv.imread(imgRight)
	grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
	grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
	
	retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
	retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
	
	if retL and retR == True:
		objpoints.append(objp)
		
		cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
		imgpointsL.append(cornersR)
		
		cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
		imgpointsR.append(cornersR)
		
		#cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
		#cv.imshow('img left', imgL)
		#cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
		#cv.imshow('img right', imgR)
		#cv.waitKey(50)
		
cv.destroyAllWindows()

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, frameSize, criteria=criteria_stereo, flags=flags)

rectifyScale = 0.2
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, frameSize, rot, trans, flags=cv.CALIB_ZERO_DISPARITY, alpha=rectifyScale)

rvecL, _ = cv.Rodrigues(rectL)
rvecL[2] = 0  # Removes roll
rvecL[0] -= 0.02 # shifts pitch
rectL, _ = cv.Rodrigues(rvecL)

rvecR, _ = cv.Rodrigues(rectR)
rvecR[2] = 0  # Removes roll
rvecR[0] += 0.03 # shifts pitch
rectR, _ = cv.Rodrigues(rvecR)


projMatrixL[0, 0] *= 0.967  # Scale down x-axis focal length
projMatrixL[1, 1] *= 0.967  # Scale down y-axis focal length


stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, frameSize, cv.CV_32FC1)

stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, frameSize, cv.CV_32FC1)

print("saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])

cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

cv_file.release()
print("Data Saved")

#!/usr/bin/python3

import cv2
import numpy as np
import os
import time

def show_undistorted_cameras(camera_id1, camera_id2):
    left_calib = np.load("./camera_params_left.npz")
    mtxL = left_calib['mtx']
    distL = left_calib['dist']
    newcamL = left_calib['newcameramtx']

    right_calib = np.load("./camera_params_right.npz")
    mtxR = right_calib['mtx']
    distR = right_calib['dist']
    newcamR = right_calib['newcameramtx']

    print("Loaded left and right camera calibration parameters:")
    print(f"Left Camera Matrix:\n{mtxL}")
    print(f"Left Distortion Coefficients:\n{distL}")
    print(f"Right Camera Matrix:\n{mtxR}")
    print(f"Right Distortion Coefficients:\n{distR}")

    output_dir = './undistorted_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap1 = cv2.VideoCapture(camera_id1, cv2.CAP_V4L2)
    cap2 = cv2.VideoCapture(camera_id2, cv2.CAP_V4L2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both cameras")
        return
    
    frame_count = 0
    total_frames = 30
    interval = 3  
    last_saved_time = time.time()

    while frame_count < total_frames:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Error: Failed to capture frames")
            break

        frame1 = cv2.flip(frame1, -1)
        frame2 = cv2.flip(frame2, -1)

        undistorted_left = cv2.undistort(frame1, mtxL, distL, None, newcamL)
        undistorted_right = cv2.undistort(frame2, mtxR, distR, None, newcamR)

        combined_frames = np.concatenate((undistorted_left, undistorted_right), axis=1)

        cv2.imshow('Undistorted Left and Right Cameras', combined_frames)

        current_time = time.time()
        if current_time - last_saved_time >= interval:
            frame_count += 1
            left_filename = f"{output_dir}/undistorted_left_{frame_count:02d}.png"
            right_filename = f"{output_dir}/undistorted_right_{frame_count:02d}.png"
            cv2.imwrite(left_filename, undistorted_left)
            cv2.imwrite(right_filename, undistorted_right)
            print(f"Saved {left_filename} and {right_filename}")
            last_saved_time = current_time


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print("\nCapture complete!")


camera_id1 = "/dev/video2"  
camera_id2 = "/dev/video0"  

if __name__ == "__main__":
    show_undistorted_cameras(camera_id1, camera_id2)

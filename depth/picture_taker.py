import cv2
import numpy as np
import threading
import os
from datetime import datetime
import time
from os import path

class Start_Cameras:

    def __init__(self, device_path):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

        self.device_path = device_path
        self.open()

    def open(self):
        try:
            self.video_capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
            grabbed, frame = self.video_capture.read()
            if not grabbed:
                raise RuntimeError("Failed to grab initial frame")
            print(f"Camera {self.device_path} is opened")
        except RuntimeError:
            self.video_capture = None
            print(f"Unable to open camera {self.device_path}")
            return
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        if self.video_capture is not None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera, daemon=True)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        if self.read_thread:
            self.read_thread.join()

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print(f"Could not read image from camera {self.device_path}")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread is not None:
            self.read_thread.join()


def TakePictures():
    total_photos = 30  
    countdown = 5  
    font = cv2.FONT_HERSHEY_SIMPLEX  

    val = input("Would you like to start the image capturing? (Y/N) ")

    if val.lower() == "y":
        left_camera = Start_Cameras("/dev/video0").start()
        right_camera = Start_Cameras("/dev/video2").start()
        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)

        counter = 0
        t2 = datetime.now()
        while counter < total_photos:
            t1 = datetime.now()
            countdown_timer = countdown - int((t1 - t2).total_seconds())

            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()
            left_frame = cv2.flip(left_frame, -1)
            right_frame = cv2.flip(right_frame, -1)

            if left_grabbed and right_grabbed:
                images = np.hstack((left_frame, right_frame))
                if countdown_timer == -1:
                    counter += 1
                    print(counter)

                    if path.isdir('../images'):
                        filename_left = f"../images/left_image_{str(counter).zfill(2)}.png"
                        filename_right = f"../images/right_image_{str(counter).zfill(2)}.png"
                        cv2.imwrite(filename_left, left_frame)
                        cv2.imwrite(filename_right, right_frame)
                        print(f"Images: {filename_left} and {filename_right} are saved!")
                    else:
                        os.makedirs("../images")
                        filename_left = f"../images/left_image_{str(counter).zfill(2)}.png"
                        filename_right = f"../images/right_image_{str(counter).zfill(2)}.png"
                        cv2.imwrite(filename_left, left_frame)
                        cv2.imwrite(filename_right, right_frame)
                        print(f"Images: {filename_left} and {filename_right} are saved!")

                    t2 = datetime.now()
                    time.sleep(1)
                    countdown_timer = 0

                cv2.putText(images, str(countdown_timer), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow("Images", images)

                k = cv2.waitKey(1) & 0xFF

                if k == ord('q'):
                    break

            else:
                print("Failed to grab frames from one or both cameras")
                break

        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
        cv2.destroyAllWindows()

    elif val.lower() == "n":
        print("Quitting!")
        exit()
    else:
        print("Please try again!")


if __name__ == "__main__":
    TakePictures()

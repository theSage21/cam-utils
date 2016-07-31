import os
import cv2
import base64
from threading import Thread

# what scripts to use
from screen_brightness import run as screen_run
from face_distance import run as face_dist_run
# function , every_n_frames
runnables = [(screen_run, 20),
             (face_dist_run, 15)]


print('python run_script.py')

#config options
display = True  # display webcam output?
#IP = 'http://192.168.11.10:8080/videofeed'  # IP camera option is available
IP = None

# -----------------------------------------------

if IP is not None:
    video_capture = cv2.VideoCapture(IP)
else:
    video_capture = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = video_capture.read()
    frame_count += 1
    for run, n_frames in runnables:
        if frame_count % n_frames == 0:
            Thread(target=run, args=(frame, )).run()
    if display:
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

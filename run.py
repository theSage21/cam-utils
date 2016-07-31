import os
import cv2

# what scripts to use
from screen_brightness import run as screen_run
from face_distance import run as face_dist_run
runnables = [screen_run, face_dist_run]


print('python run_script.py')

#config options
display = True  # display webcam output?

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    for run in runnables:
        run(frame)
    if display:
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

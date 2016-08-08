# -----------------------------------------------
# import the things we are going to need
import json
import cv2
from threading import Thread

# -----------------------------------------------
# import scripts to be used
from attendance import run as attendance_run
from screen_brightness import run as screen_run
from face_distance import run as face_dist_run

# function , every_n_frames
runnables = [#(screen_run, 50),
             #(face_dist_run, 50),
             (attendance_run, 3)]
# -----------------------------------------------
# Load configuration files
print('python run_script.py| config.json is for configuration')
config = json.loads(open('config.json', 'r').read())

# -----------------------------------------------
# run program

# Choose input device
if config['cam'] == 'ip':
    video_capture = cv2.VideoCapture(config['ip'])
elif config['cam'] == 'inbuilt':
    video_capture = cv2.VideoCapture(0)

# get frames and run scripts on them
frame_count = 0
while True:
    ret, frame = video_capture.read()
    frame_count += 1
    for run, n_frames in runnables:
        if frame_count % n_frames == 0:
            Thread(target=run, args=(frame, )).run()
    if config['show_video']:
        cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

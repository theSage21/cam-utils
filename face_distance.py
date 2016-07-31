import os
import cv2

#config options
casc = 'support/haarcascade_frontalface_default.xml'
notification_command = "notify-send -u critical 'MOVE Back'"

def is_too_close(frame):
    "Is the person too close to the camera?"
    global casc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(casc)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # NOTE: We try to see if I'm sitting too close
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_frame = frame[y: y + h, x: x + w]
        if (cropped_frame.size) > 196573:
            return True
    return False

def notify():
    "Send a notification to th erespective backend"
    global notification_command
    os.system(notification_command)

def run(frame):
    if is_too_close(frame):
        notify()

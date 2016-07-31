import os
import cv2

casc = 'support/haarcascade_frontalface_default.xml'
face_folder = 'storage/faces/'

def get_faces_in_frame(frame):
    "Get the cropped faces in the frame"
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
    face_images = [frame[y: y + h, x: x + w].copy() for x, y, w, h in faces]
    return face_images

def get_known_faces():
    "return a list of known face names"
    global face_folder
    return os.listdir(face_folder)

def get_classifier():
    "Returns a classifier for classifying faces"
    # TODO
    return None

def classify_face(face):
    "Given a face, classify it as one of many known faces"
    # TODO
    return None

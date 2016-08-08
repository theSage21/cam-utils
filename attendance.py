import os
import cv2
import numpy as np
from PIL import Image
from scipy.misc import imsave

casc = 'support/haarcascade_frontalface_default.xml'
face_folder = 'storage/attendance/faces/'

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
    "return a list of known face images and their names"
    global face_folder, casc
    faceCascade = cv2.CascadeClassifier(casc)
    folders = os.listdir(face_folder)
    images, labels, label_map = [], [], {}
    for index, person in enumerate(folders):
        label_map[index] = person.replace('_',' ').title()
        imgs = os.listdir(face_folder + '/' + person)
        for im in imgs:
            i = os.path.join(face_folder, person, im)
            image_pil = Image.open(i).convert('L')
            image_np = np.array(image_pil, 'uint8')
            detected_faces = faceCascade.detectMultiScale(image_np)
            for (x, y, w, h) in detected_faces:
                images.append(image_np[y: y + h, x: x + w])
                labels.append(index)
    return images, labels, label_map

def get_classifier():
    "Returns a classifier for classifying faces"
    rec = cv2.face.createLBPHFaceRecognizer()
    return rec

def train_classifier(images, labels):
    "Given face image paths and labels: train a classifier"
    cl = get_classifier()
    cl.train(images, np.array(labels))
    return cl

def label_faces(frame, recognizer, label_map, fontsize=0.8):
    "Label all known faces in the image"
    global casc
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(casc)
    predict_image = np.array(image, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        face = predict_image[y: y + h, x: x + w]
        return_value = recognizer.predict(face)
        cv2.putText(frame, label_map[return_value], (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontsize, 255)
        cv2.rectangle(frame, (x, y + 5), (x+w, y+h), (90, 90, 90), 2)
    return frame

def test():
    if ret:
        image = label_faces(frame, cl, label_map)
        imsave('my_image.png', image)
    video_capture.release()
    cv2.destroyAllWindows()

def test():
    faces, labels, label_map = get_known_faces()
    cl = train_classifier(faces, labels)
    video_capture = cv2.VideoCapture(0)
    print('Running loop')
    while True:
        ret, frame = video_capture.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_image = label_faces(frame, cl, label_map)
        cv2.imshow('Video', new_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

test()

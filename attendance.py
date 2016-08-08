import os
import cv2
import pickle
import numpy as np
from PIL import Image
from scipy.misc import imsave

casc = 'support/haarcascade_frontalface_default.xml'
face_folder = 'storage/attendance/faces/'
recognizer_path = 'storage/attendance/recognizer'
label_map_path = 'storage/attendance/recognizer_label_map'

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
    global recognizer_path
    rec = cv2.face.createLBPHFaceRecognizer()
    from_disk = os.path.exists(recognizer_path)
    if from_disk:
        rec.load(recognizer_path)
    return rec, from_disk

def train_classifier():
    "Given face image paths and labels: train a classifier"
    global recognizer_path, label_map_path
    cl, from_disk = get_classifier()
    if not from_disk:
        print('Training recognizer')
        images, labels, label_map = get_known_faces()
        cl.train(images, np.array(labels))
        cl.save(recognizer_path)
        pickle.dump(label_map, open(label_map_path, 'wb'))
    else:
        label_map = pickle.load(open(label_map_path, 'rb'))
    return cl, label_map

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

def run(frame):
    cl, label_map = train_classifier()
    new_image = label_faces(frame, cl, label_map)
    cv2.imshow('attendance_window', new_image)
    return new_image


def test():
    '''
    test this app
    '''
    faces, labels, label_map = get_known_faces()
    cl, label_map = train_classifier()
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


if __name__ == '__main__':
    test()

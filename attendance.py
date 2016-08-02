import os
import cv2
from PIL import Image

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
            image_pil = Image.open(image_path).convert('L')
            image_np = np.array(image_pil, 'uint8')
            detected_faces = faceCascade.detectMultiScale(image_np)
            for (x, y, w, h) in detected_faces:
                images.append(image_np[y: y + h, x: x + w])
                labels.append(index)
            images.append(i_data)
    return images, labels, label_map

def get_classifier():
    "Returns a classifier for classifying faces"
    rec = cv2.face.craeteLBPHFaceRecognizer()
    return rec

def train_classifier(face_data, labels):
    "Given face image paths and labels: train a classifier"
    cl = get_classifier()
    cl.train(images, np.array(labels))

def label_faces(image, recognizer, label_map):
    "Label all known faces in the image"
    global casc
    predict_image = np.array(predict_image_pil, 'uint8')
    faceCascade = cv2.CascadeClassifier(casc)
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        cv2.putText(image, label_map[nbr_predicted], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    return image

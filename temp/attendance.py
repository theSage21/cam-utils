import os
import cv2
import pickle
import numpy as np
from PIL import Image
from scipy.misc import imsave, imresize, imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from skimage.feature import hog
from skimage import data, color, exposure
print('Imports done')

#  - - - - - - - - - config things and global things
casc = '../support/haarcascade_frontalface_default.xml'
face_folder = '../storage/attendance/faces/'
recognizer_path = '../storage/attendance/recognizer'
label_map_path = '../storage/attendance/recognizer_label_map'
face_size = (200, 200)
faceCascade = cv2.CascadeClassifier(casc)
hog = cv2.HOGDescriptor()
winStride, padding, locations= (8,8), (8,8), ((10,20),)
test_image = 'hero.jpg'
minNeighbors = 15
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# - - - - - - - - - function definitions
def get_known_faces():
    "return a list of known face images and their names"
    global face_folder, face_size, faceCascade, minNeighbors
    folders = os.listdir(face_folder)
    images, labels, label_map = [], [], {}
    for index, person in enumerate(folders):
        label_map[index] = person.replace('_',' ').title()
        imgs = os.listdir(face_folder + '/' + person)
        for im in imgs:
            i = os.path.join(face_folder, person, im)
            image_pil = Image.open(i).convert('L')
            image_np = np.array(image_pil, 'uint8')
            detected_faces = faceCascade.detectMultiScale(image_np, minNeighbors=minNeighbors)
            for (x, y, w, h) in detected_faces:
                face_of_unknown_size = image_np[y: y + h, x: x + w]
                normalized_face = imresize(face_of_unknown_size, face_size)
                images.append(normalized_face)
                labels.append(index)
    return images, labels, label_map

# - - - - - - - - - compute


image_pil = imread(test_image)
image = cv2.cvtColor(image_pil, cv2.COLOR_BGR2GRAY)

detected_faces = faceCascade.detectMultiScale(image, minNeighbors=minNeighbors)
# lots of false positives, need to clean
# done by increasing minNeighbors

frame = image
for x, y, w, h in detected_faces:
    frame = cv2.rectangle(frame, (x, y + 5), (x+w, y+h), (0, 0, 0), 2)
imsave('detected.jpg', frame)
print('Detected faces. Total of {} found'.format(len(detected_faces)))

# Label those faces
images, labels, label_map = get_known_faces()
x, y = [], []
print('{} samples available in dataset'.format(len(images)))
for image, label in zip(images, labels):
    hist = hog.compute(image, winStride, padding, locations)[:, 0]
    x.append(hist)
    y.append(label)

scores = cross_val_score(rf, x, y)
print(scores.mean(), scores)

# ---------------------------------------------------------------------
def test_cam():
    '''
    test this app
    '''
    cl, label_map = train_classifier()
    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture("http://192.168.43.1:8080/videofeed")
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

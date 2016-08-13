import os
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imsave, imresize, imread
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from skimage.feature import hog
from skimage import data, color, exposure
print('Imports done')

#  - - - - - - - - - config things and global things
casc = '../support/haarcascade_frontalface_default.xml'
face_folder = '../storage/attendance/faces/'
recognizer_path = '../storage/attendance/recognizer'
label_map_path = '../storage/attendance/recognizer_label_map'
face_size = (100, 100)
faceCascade = cv2.CascadeClassifier(casc)
hog = cv2.HOGDescriptor()
winStride, padding, locations= (8,8), (8,8), ((10,20),)
test_image = 'hero.jpg'
minNeighbors = 5
n_components = 150  # for eigenfaces
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5, 1],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

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
                images.append(normalized_face.flatten())
                labels.append(index)
    return images, labels, label_map

# - - - - - - - - - train classifiers

# Label those faces
minNeighbors = 5
images, labels, label_map = get_known_faces()
h, w = face_size
n_samples = len(images)

x, y = [], []
print('{} samples available in dataset. {}x{} images'.format(n_samples, h, w))
for image, label in zip(images, labels):
    #hist = hog.compute(image, winStride, padding, locations)[:, 0]
    #x.append(hist)
    x.append(image)
    y.append(label)
x, y = np.array(x), np.array(y)

for lab in set(y):
    print('{} samples for {}'.format(labels.count(lab), label_map[lab]))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# EIGENFACES

print("Extracting the top %d eigenfaces from %d faces"
              % (n_components, X_train.shape[0]))
print(X_train.shape)
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
#eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# TRAIN SVM

print("Fitting the classifier to the training set")
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf = KNeighborsClassifier(n_neighbors=2, weights='distance', n_jobs=-1)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")

# PREDICT CROSSVAL

print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(confusion_matrix(y_test, y_pred))

# ------------------------------test on a new image: hero.jpg
image_pil = imread(test_image)
image = cv2.cvtColor(image_pil, cv2.COLOR_BGR2GRAY)

minNeighbors = 15
detected_faces = faceCascade.detectMultiScale(image, minNeighbors=minNeighbors)
# lots of false positives, need to clean
# done by increasing minNeighbors

frame = image
for x, y, w, h in detected_faces:
    frame = cv2.rectangle(frame, (x, y + 5), (x+w, y+h), (0, 0, 0), 2)
imsave('detected.jpg', frame)
print('Detected faces. Total of {} found'.format(len(detected_faces)))




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

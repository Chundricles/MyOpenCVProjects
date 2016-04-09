import cv2, os
import numpy as np
from PIL import Image

faceCascade = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
recognizer = cv2.createLBPHFaceRecognizer()



def get_images_and_labels(path):
    i=1
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
            print i
            i=i+1
        return images, labels





images, labels = get_images_and_labels('C:/Users/Neil/Pictures/SamplesUsedForOpenCV/att_faces/s1')

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
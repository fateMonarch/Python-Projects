# -*- coding: utf-8 -*-

import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
dataset = "C:/Users/muham/.spyder-py3/dataset" 

for filename in os.listdir(dataset):
    if filename.endswith(".jpg"):

        img_path = os.path.join(dataset, filename)
        img = cv2.imread(img_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Yuz Tanima', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
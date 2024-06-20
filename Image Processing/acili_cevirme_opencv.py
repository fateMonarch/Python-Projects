# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    height, width, channels = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
rotated_img = rotate_image(img, 330)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Orijinal Resim")

plt.subplot(1, 2, 2)
plt.imshow(rotated_img)
plt.title("30 Derece")
plt.show()

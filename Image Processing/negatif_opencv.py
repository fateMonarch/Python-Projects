# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def negatifles(img):
    negatif_img = cv2.bitwise_not(img)
    return negatif_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
negatif_img = negatifles(img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(negatif_img, cmap="gray")
plt.title("Negatif")
plt.show()
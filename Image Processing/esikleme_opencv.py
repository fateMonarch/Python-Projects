# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def esikle(img, esikle):
    gri_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, esiklenmis_img = cv2.threshold(gri_img, esikle, 255, cv2.THRESH_BINARY)
    return esiklenmis_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
esikleme_degeri = 128
esiklenmis_img = esikle(img, esikleme_degeri)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(esiklenmis_img, cmap='gray')
plt.title("Esiklenmis Resim")
plt.show()

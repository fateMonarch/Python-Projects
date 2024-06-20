# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def parlaklik_ayarla(img, alpha, beta):
    parlakligi_ayarlanmis_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return parlakligi_ayarlanmis_img

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
alpha, beta = 1, 50
parlak_img = parlaklik_ayarla(img, alpha, beta)
alpha, beta = 1, -50
sonuk_img = parlaklik_ayarla(img, alpha, beta)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(parlak_img, cv2.COLOR_BGR2RGB))
plt.title("Parlak")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(sonuk_img, cv2.COLOR_BGR2RGB))
plt.title("Sonuk")
plt.show()

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def parlaklik_ayarla(img, alpha, beta):
    parlakligi_ayarlanmis_img = np.clip(img * parlaklik_ayari, 0, 255).astype(np.uint8)
    return parlakligi_ayarlanmis_img

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')

alpha, beta, parlaklik_ayari = 1, 0, 1.5
parlak_img = parlaklik_ayarla(img, alpha, beta)
alpha, beta, parlaklik_ayari = 1, 0, 0.5
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

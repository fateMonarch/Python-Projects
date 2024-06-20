# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
kaydirma = 150
height, width, _ = img.shape

saga_otelenmis_img = np.zeros_like(img)
rows, cols, _ = img.shape
M = np.float32([[1, 0, kaydirma], [0, 1, 0]])
saga_otelenmis_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_WRAP)

sola_otelenmis_img = np.zeros_like(img)
M = np.float32([[1, 0, -kaydirma], [0, 1, 0]])
sola_otelenmis_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_WRAP)

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Orijinal")

plt.subplot(1, 3, 2)
plt.imshow(saga_otelenmis_img)
plt.title("Sağa Otelenmiş")

plt.subplot(1, 3, 3)
plt.imshow(sola_otelenmis_img)
plt.title("Sola Otelenmiş")
plt.show()

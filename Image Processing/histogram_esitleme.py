# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def esitleme(gri_img):
    histogram = np.zeros(256, dtype=int)
    for i in range(gri_img.shape[0]):
        for j in range(gri_img.shape[1]):
            histogram[gri_img[i, j]] += 1
            
    dagılım = np.cumsum(histogram)
    esitlenmis_img = (dagılım[gri_img] * 255 / dagılım[-1]).astype(np.uint8)
    return esitlenmis_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gri_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
esitlenmis_img = esitleme(gri_img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(esitlenmis_img, cmap='gray')
plt.title("Histogram Esitleme")
plt.show()

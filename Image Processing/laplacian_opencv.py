# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_filtresi(img):
    laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_img = np.clip(np.abs(laplacian_img), 0, 255).astype(np.uint8)
    return laplacian_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
laplacian_img = laplacian_filtresi(img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(laplacian_img, cmap="gray")
plt.title("Laplacian")
plt.show()

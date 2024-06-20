# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)

kernel_size = 1
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
yayilmis_img = np.zeros_like(img)
rows, cols = img.shape

for i in range(rows):
    for j in range(cols):
        if img[i, j] > 0:
            yayilmis_img[i:i+kernel_size, j:j+kernel_size] = 255

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(yayilmis_img, cmap='gray')
plt.title("Yayma (Dilation)")
plt.show()

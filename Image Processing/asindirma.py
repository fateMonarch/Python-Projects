# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

rows, cols = img.shape
asinmis_img = np.zeros_like(img)

kernel_rows, kernel_cols = kernel.shape
offset_rows, offset_cols = kernel_rows // 2, kernel_cols // 2

for i in range(offset_rows, rows - offset_rows):
    for j in range(offset_cols, cols - offset_cols):
        roi = img[i - offset_rows:i + offset_rows + 1, j - offset_cols:j + offset_cols + 1]
        asinmis_img[i, j] = np.min(roi * kernel)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(asinmis_img, cmap='gray')
plt.title("Asindirma (Erosion)")
plt.show()

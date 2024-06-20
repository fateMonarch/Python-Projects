# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
asinmis_img = cv2.erode(img, kernel, iterations=1)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(asinmis_img, cmap='gray')
plt.title("Asindirma (Erosion)")
plt.show()

# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
yayilmis_img = cv2.dilate(img, kernel, iterations=1)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(yayilmis_img, cmap='gray')
plt.title("Yayma (Dilation)")
plt.show()

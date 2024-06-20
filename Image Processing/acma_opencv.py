# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)

asindirma_kernel_size = 5
asindirma_kernel = np.ones((asindirma_kernel_size, asindirma_kernel_size), dtype=np.uint8)
yayma_kernel_size = 5
yayma_kernel = np.ones((yayma_kernel_size, yayma_kernel_size), dtype=np.uint8)

acma_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, asindirma_kernel)
acma_img = cv2.dilate(acma_img, yayma_kernel, iterations=1)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(acma_img, cmap='gray')
plt.title("Acma (Opening)")
plt.show()

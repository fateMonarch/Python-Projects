# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def kapama(img, asindirma_kernel, yayma_kernel):
    yayma_img = cv2.dilate(img, yayma_kernel, iterations=1)
    asindirma_img = cv2.erode(yayma_img, asindirma_kernel, iterations=1) 
    return asindirma_img

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)
yayma_kernel_size = 5    
yayma_kernel = np.ones((yayma_kernel_size, yayma_kernel_size), dtype=np.uint8)

asindirma_kernel_size = 5
asindirma_kernel = np.ones((asindirma_kernel_size, asindirma_kernel_size), dtype=np.uint8)

_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kapama_img = kapama(binary_img, asindirma_kernel, yayma_kernel)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(kapama_img, cmap='gray')
plt.title("Kapama (Closing)")
plt.show()

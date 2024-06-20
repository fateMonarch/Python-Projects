# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def kapama_yap(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kapama_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return kapama_img

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg', cv2.IMREAD_GRAYSCALE)
kernel_size = 5
kapama_img = kapama_yap(img, kernel_size)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(kapama_img, cmap='gray')
plt.title("Kapama (Closing)")
plt.show()

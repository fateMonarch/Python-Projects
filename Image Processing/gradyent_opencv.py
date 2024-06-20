# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gradyent(img):
    gri_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gri_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gri_img, cv2.CV_64F, 0, 1, ksize=3)
    
    gradyent = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradyent, sobel_x, sobel_y

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gradyent, sobel_x, sobel_y = gradyent(img)

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(2, 2, 2)
plt.imshow(gradyent, cmap='gray')
plt.title('Gradyent')

plt.subplot(2, 2, 3)
plt.imshow(sobel_x, cmap='gray')

plt.subplot(2, 2, 4)
plt.imshow(sobel_y, cmap='gray')
plt.show()
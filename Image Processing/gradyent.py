# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def gradyent(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gri_img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    sobel_x = convolve(gri_img, kernel_x)
    sobel_y = convolve(gri_img, kernel_y)
    
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
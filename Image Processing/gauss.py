# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def gauss_kernel(size, sigma=1.0):
    if sigma == 0:
        return np.ones((size, size)) / (size * size)
    else:
        kernel = np.fromfunction(
            lambda x, y: (1/ (2*np.pi*sigma**2)) * 
                         np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

def gauss_filtresi(img, kernel_size=(5, 5), sigma=1.0):
    kernel = gauss_kernel(kernel_size[0], sigma)
    gauss_img = convolve(img, kernel[..., np.newaxis])
    return gauss_img.astype(np.uint8)

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gauss_img = gauss_filtresi(img, kernel_size=(5, 5), sigma=0)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(gauss_img, cv2.COLOR_BGR2RGB))
plt.title('Gauss')
plt.show()

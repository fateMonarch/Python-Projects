# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def esikleme(img, threshold):
    height, width = img.shape
    binary_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            if pixel_value > threshold:
                binary_img[i, j] = 255
            else:
                binary_img[i, j] = 0

    return binary_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gri_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])

threshold_value = 127
binary_img = esikleme(gri_img, threshold_value)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(binary_img, cmap='gray')
plt.title('Esiklenmis Resim')
plt.show()
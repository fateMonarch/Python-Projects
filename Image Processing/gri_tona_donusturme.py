# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def griles(img):
    height, width, _ = img.shape
    gri_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            gri_value = int(0.299 * img[i, j, 2] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 0])
            gri_img[i, j] = gri_value

    return gri_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gri_img = griles(img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(gri_img, cmap='gray')
plt.title("Gri Tonlu Resim")
plt.show()

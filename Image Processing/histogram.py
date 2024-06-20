# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def histogram(img):
    height, width = img.shape
    hist = np.zeros(256, dtype=int)

    for i in range(height):
        for j in range(width):
            pixel_value = int(img[i, j])
            hist[pixel_value] += 1

    return hist

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gri_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
hist = histogram(gri_img)

plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Piksel Degerleri')
plt.ylabel('Piksel Sayisi')
plt.show()
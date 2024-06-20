# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
kaydırma = 150
height, width, _ = img.shape
saga_otelenmis_img = np.zeros_like(img)
sola_otelenmis_img = np.zeros_like(img)

for i in range(height):
    for j in range(width):
        new_j = (j + kaydırma) % width
        saga_otelenmis_img[i, new_j, :] = img[i, j, :]
        
for i in range(height):
    for j in range(width):
        new_j = (j - kaydırma) % width
        sola_otelenmis_img[i, new_j, :] = img[i, j, :]

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(saga_otelenmis_img)
plt.title("Saga Otelenmis")

plt.subplot(1, 3, 3)
plt.imshow(sola_otelenmis_img)
plt.title("Sola Otelenmis")
plt.show()

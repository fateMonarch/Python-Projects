# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def laplacian_filtresi(img):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    
    laplacian_img = np.zeros_like(img)
    for c in range(img.shape[2]):
        laplacian_img[..., c] = np.clip(np.abs(np.convolve(img[..., c].flatten(), 
                                                           kernel.flatten(), mode='same')), 0, 255).reshape(img.shape[:-1])
  
    return laplacian_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
laplacian_img = laplacian_filtresi(img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(laplacian_img, cmap="gray")
plt.title("Laplacian")
plt.show()

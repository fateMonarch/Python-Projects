# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def negatifles(img):
    negatif_img = 255 - img
    return negatif_img

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
negatif_img = negatifles(img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(negatif_img, cmap="gray")
plt.title("Negatif")
plt.show()
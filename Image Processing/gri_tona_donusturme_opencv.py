# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gri_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(gri_img, cmap='gray')
plt.title("Gri Tonlu Resim")
plt.show()

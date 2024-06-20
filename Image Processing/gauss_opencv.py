# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def gauss_filtresi(img, kernel_size=(5, 5), sigma=0):
    gauss_img = cv2.GaussianBlur(img, kernel_size, sigma)
    return gauss_img

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
gauss_img = gauss_filtresi(img, kernel_size=(5, 5), sigma=0)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(gauss_img, cv2.COLOR_BGR2RGB))
plt.title('Gauss Filtresi')
plt.show()

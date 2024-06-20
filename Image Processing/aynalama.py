# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
height, width, _ = img.shape

x_eksenine_gore_img = img[::-1, :, :]
y_eksenine_gore_img = img[:, ::-1, :]
    
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(x_eksenine_gore_img)
plt.title("X Eksenine Gore Aynalama")

plt.subplot(1, 3, 3)
plt.imshow(y_eksenine_gore_img)
plt.title("Y Eksenine Gore Aynalama")
plt.show()

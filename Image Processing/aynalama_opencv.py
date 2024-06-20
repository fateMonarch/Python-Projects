# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
x_eksenine_gore_img = np.flipud(img)
y_eksenine_gore_img = np.fliplr(img)

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

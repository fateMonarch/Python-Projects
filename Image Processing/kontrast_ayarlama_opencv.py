# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def kontrast_ayarla(img, alpha, beta):
    kontrastı_ayarlanmıs_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return kontrastı_ayarlanmıs_img

img = cv2.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
alpha, beta = 1.5, 30
kontrastı_arttirilmis_img = kontrast_ayarla(img, alpha, beta)
alpha, beta = 0.5, 30
kontrastı_azaltilmis_img = kontrast_ayarla(img, alpha, beta)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(kontrastı_arttirilmis_img, cv2.COLOR_BGR2RGB))
plt.title("Cok Kontrast")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(kontrastı_azaltilmis_img, cv2.COLOR_BGR2RGB))
plt.title("Az Kontrast")
plt.show()

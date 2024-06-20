# -*- coding: utf-8 -*-

import cv2
import numpy as np

square_top_left = (-1, -1)
square_size = -1

def kare_ciz(event, x, y, flags, param):
    global square_top_left, square_size, img
    if event == cv2.EVENT_LBUTTONDOWN:
        square_top_left = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        square_size = max(x - square_top_left[0], y - square_top_left[1])
        roi = img[square_top_left[1]:square_top_left[1] + square_size, square_top_left[0]:square_top_left[0] + square_size]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gri_img, roi_gray, cv2.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + square_size, pt[1] + square_size), (0, 255, 0), 2)

        cv2.imshow("Korelasyon", img)

img = cv2.imread('C:/Users/muham/.spyder-py3/EA.png')
gri_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if img is None:
    print("Görüntü yüklenemedi.")
else:
    cv2.namedWindow("Korelasyon")
    cv2.setMouseCallback("Korelasyon", kare_ciz)

    while True:
        cv2.imshow("Korelasyon", img)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

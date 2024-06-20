# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math

def rotate_image(image, angle):
    height, width, channels = image.shape
    theta = math.radians(angle)

    rotation_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

    expanded_rotation_matrix = np.hstack([rotation_matrix, np.zeros((2, 1))])
    expanded_rotation_matrix = np.vstack([expanded_rotation_matrix, [0, 0, 1]])
    rotated_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            rotated_coords = np.dot(expanded_rotation_matrix, [i, j, 1])
            rotated_i, rotated_j = rotated_coords[:2].astype(int)
            
            if 0 <= rotated_i < height and 0 <= rotated_j < width:
                rotated_image[rotated_i, rotated_j, :] = image[i, j, :]

    return rotated_image

img = plt.imread('C:/Users/muham/.spyder-py3/Jojo.jpeg')
rotated_img = rotate_image(img, 330)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Orijinal Resim")

plt.subplot(1, 2, 2)
plt.imshow(rotated_img)
plt.title("30 Derece Döndürülmüş Resim")
plt.show()

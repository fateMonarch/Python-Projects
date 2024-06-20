# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

islenmis_resim_dizini = 'C:/Users/muham/.spyder-py3/İslenmis'
islenmis_resimler = [dosya for dosya in os.listdir(islenmis_resim_dizini) if dosya.endswith(('.png', '.jpg', '.jpeg'))]

num_classes = 40
class_names = ['Original', 'Rotated Original', 'Slightly Ripped', 'Rotated Slightly Ripped', 'Very Ripped', 'Rotated Very Ripped',
               'Slightly Banded', 'Rotated Slightly Banded', 'Very Banded', 'Rotated Very Banded', 'Slightly Drawn In Pencil',
               'Rotated Slightly Drawn In Pencil', 'Very Drawn In Pencil', 'Rotated Very Drawn In Pencil', 'Slightly Drawn In Pen',
               'Rotated Slightly Drawn In Pen', 'Very Drawn In Pen', 'Rotated Very Drawn In Pen', '1', '2', '5', '10', '20', '50',
               '100', '200', '500', 'TL', 'USD', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']

def veri_seti_olustur(islenmis_resimler):
    X, y = [], []

    for dosya_ad in islenmis_resimler:
        dosya_yolu = os.path.join(islenmis_resim_dizini, dosya_ad)
        try:
            with open(dosya_yolu, 'rb') as f:
                dosya_icerik = f.read()
                nparr = np.frombuffer(dosya_icerik, np.uint8)
                resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if resim is not None:
                resim = cv2.resize(resim, (224, 224))
                resim = resim / 255.0

                X.append(resim)
                y.append(dosya_ad.split('_')[0])
            else:
                print(f"'{dosya_ad}' okunamadı!")

        except Exception as e:
            print(f"'{dosya_ad}' okunamadı ve hata kodu: {str(e)}")

    y_encoded = model.fit_transform(y)
    return np.array(X), np.array(y_encoded)

def model_olustur(num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modeli_egit(islenmis_resimler):
    X, y = veri_seti_olustur(islenmis_resimler)
    X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_egitim, X_dogrulama, y_egitim, y_dogrulama = train_test_split(X_egitim, y_egitim, test_size=0.1, random_state=42)

    model = model_olustur(num_classes)

    y_egitim = keras.utils.to_categorical(y_egitim, num_classes)
    y_dogrulama = keras.utils.to_categorical(y_dogrulama, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.fit(X_egitim, y_egitim, epochs=10, validation_data=(X_dogrulama, y_dogrulama)) 
    loss, accuracy = model.evaluate(X_test, y_test)
    return model

model = modeli_egit(islenmis_resimler)
model.save('proje-modeli-3.keras')

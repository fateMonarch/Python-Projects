# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

veri = pd.read_csv('C:\\Users\\muham\\OneDrive\\Masaüstü\\D\\5\\Yapay Zeka Ve Uzman Sistemler\\chronic_kidney_disease.xlsx')
X = veri.drop('age', axis=1)  
y = veri['class']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 3
k_nn_model = KNeighborsClassifier(n_neighbors=k)
k_nn_model.fit(X_train, y_train)

y_pred = k_nn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)   
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Doğruluk: ", accuracy)
print("Sınıflandırma Raporu:\n", report)
print("Karışıklık Matrisi:\n", confusion)

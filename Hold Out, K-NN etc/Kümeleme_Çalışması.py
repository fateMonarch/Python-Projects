# -*- coding: utf-8 -*-

import warnings
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
veri = pd.read_excel("C:/Users/muham/.spyder-py3/chronic_kidney_disease.xlsx")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)

x = veri.iloc[:, :-1]
y = veri.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_pred_kfold = cross_val_predict(knn_model, x, y, cv=kf)
conf_matrix_kfold = confusion_matrix(y, y_pred_kfold)
class_report_kfold = classification_report(y, y_pred_kfold)
accuracy_kfold = cross_val_score(knn_model, x, y, cv=kf, scoring='accuracy')

print("\nK-Fold Çapraz Doğrulama Sonuçları (Kümelenmemiş Veri):")
print("Doğruluklar:", accuracy_kfold)
print("Ortalama Doğruluk:", accuracy_kfold.mean())
print("Karışıklık Matrisi:")
print(conf_matrix_kfold)
print("\nSınıflandırma Raporu:")
print(class_report_kfold)

kmeans = KMeans(n_clusters=3, random_state=42)
veri['Kume'] = kmeans.fit_predict(x)

x_kume = veri.iloc[:, :-1]
y_kume = veri.iloc[:, -1]
x_kume_train, x_kume_test, y_kume_train, y_kume_test = train_test_split(x_kume, y_kume, test_size=0.2, random_state=42)

y_kume_pred_kfold = cross_val_predict(knn_model, x_kume, y_kume, cv=kf)
conf_matrix_kfold_kume = confusion_matrix(y_kume, y_kume_pred_kfold)
class_report_kfold_kume = classification_report(y_kume, y_kume_pred_kfold)
accuracy_kfold_kume = cross_val_score(knn_model, x_kume, y_kume, cv=kf, scoring='accuracy')

print("\nK-Fold Çapraz Doğrulama Sonuçları (Kümelenmiş Veri):")
print("Doğruluklar:", accuracy_kfold_kume)
print("Ortalama Doğruluk:", accuracy_kfold_kume.mean())
print("Karışıklık Matrisi:")
print(conf_matrix_kfold_kume)
print("\nSınıflandırma Raporu:")
print(class_report_kfold_kume)

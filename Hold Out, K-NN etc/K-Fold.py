# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
import time

veri = "C:\\Users\\muham\\OneDrive\\Masaüstü\\D\\5\\Yapay Zeka Ve Uzman Sistemler\\chronic_kidney_disease.xlsx"

veriler = pd.read_excel(veri)
etiketler = [1] * 158
i= 1

while i<158:
    etiketler[i]=i
    i = i+1

k_folds = 5
model = RandomForestClassifier()

for fold in range(k_folds):
    
    start_time = time.time()
    
    x_train, x_test, y_train, y_test = train_test_split(veriler, etiketler, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    
    cm = confusion_matrix(y_test, y_pred) 
    accuracy = accuracy_score(y_test, y_pred)
    
    sensitivity = recall_score(y_test, y_pred, average='micro')
    specificity = recall_score(y_test, y_pred, pos_label=0, average='micro')

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Fold {fold + 1}:")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Elapsed Time (s): {elapsed_time:.2f}\n")

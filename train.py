#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:29:35 2023

@author: aliogulcanet
"""

import pandas as pd
import numpy as np
df = pd.read_csv("heart.csv.xls")
x = df.drop(["trestbps","chol","fbs","restecg","target"],axis=1)
print(df.corr())
y = df["target"]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(x)
X = pd.DataFrame(X, columns=x.columns)
X.head()
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

print("Random Forest")
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
cm_rf=confusion_matrix(y_test,rf_pred)
rf_acc = classification_report(y_test, rf_pred)
print(rf_acc)
print("-"*60)

print(rf.predict([[23, 1, 2, 100, 0, 1, 0, 0, 0]]))
print(rf.predict([[68, 1, 3, 120, 2, 2, 3, 3, 3]]))
print(rf.predict([[58, 0, 0, 122, 0, 1, 1, 0, 2]]))
print(rf.predict([[20, 1, 2, 80, 0, 1, 0, 0, 0]]))

import pickle   
pickle.dump(rf,open("model.pkl","wb")) 
model = pickle.load(open("model.pkl","rb"))















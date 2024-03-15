# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:54:26 2024

@author: youss
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('Iris.csv')
print(df)

df.describe()

data = df.values
X = data[:, 1:4]
Y = data[:, 5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions) * 100
print('Accuracy: %.3f ' % accuracy)


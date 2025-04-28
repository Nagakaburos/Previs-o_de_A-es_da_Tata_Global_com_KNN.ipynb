# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

!pip install quandl
import quandl

data = pd.read_csv('NSE-TATAGLOBAL11.csv')

data.head(10)

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

data['open - close'] = data['Open'] - data['Close']
data['high - low'] = data['High'] - data['Low']
data = data.dropna()

X = data[['open - close', 'high - low']]
#y = data['Close']
X.head()

Y= np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

Y

from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 44)

# from sklearn.linear_model import kNeighborsClassifier # kNeighborsClassifier is not in sklearn.linear_model
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier from sklearn.neighbors instead
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

model.fit(x_train, y_train) # Changed X_train, Y_train to x_train, y_train to match variable names defined earlier

accuracy_train = accuracy_score(y_train, model.predict(x_train)) # Changed Y_train to y_train to match variable names
accuracy_test = accuracy_score(y_test, model.predict(x_test)) # Changed Y_test to y_test to match variable names

print ('train_data Accuracy: %.2f' %accuracy_train)
print ('test_data Accuracy: %.2f' %accuracy_test)

predictions_classification = model.predict(x_test)

actual_predicted_data = pd.DataFrame({'Actual Class': y_test, 'Predicted.Class': predictions_classification})

actual_predicted_data.head(10)

y = data['Close']

y

from sklearn.neighbors import KNeighborsRegressor
from sklearn import  neighbors

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state=44)

params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)

model_reg.fit(x_train_reg, y_train_reg)
predictions = model_reg.predict(x_test_reg)

print(predictions)

rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
rms

valid = pd.DataFrame({'Actual Close': y_test_reg, 'Predicted Close value': predictions})

valid.head(10)

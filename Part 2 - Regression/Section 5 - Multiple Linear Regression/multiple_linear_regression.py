# multiple linear regression

# importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset

dataset =pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# encoding the categorical data

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features =[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the dumy variable trap
X=X[:,1:]

#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size=0.2 , random_state =0)

# fitting the multiple regression model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train , y_train)

# prediction of results

y_pred= regressor.predict(X_test)

#plotting the y_pred and y_orignal

plt.plot(y_test,color='red')
plt.plot(y_pred, color='green')





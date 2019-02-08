# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
df=pd.DataFrame(X)
y= dataset.iloc[:,-1].values
dy=pd.DataFrame(y)

#missing data

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


#changing categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
onehotencoder= OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)



# polynomial regresstion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#splitting the dataset
#no splitting full dataset fro training

#fitting linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)



# fitting ploynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree =3)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualising the linear model
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='green')

#visualising the polynomial model
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='green')
plt.show()






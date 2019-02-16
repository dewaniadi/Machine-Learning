
#data preprocessing 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the dataset
dataset= pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

# splitting 
#NO SPLITTING AS THE DATA IS SMALL

#Feature Scaleing
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)



# fitting SVR to the data
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#predicting the result
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# plotting the data
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='green')


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





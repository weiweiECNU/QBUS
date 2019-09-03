
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

car_data_df = pd.read_csv('Lecture2_Regression.csv')

dim= car_data_df.shape
n_row= dim[0]
n_col= dim[1]
#car_price= car_data.iloc[:,1]

car_price= car_data_df['Price']
odometer= car_data_df['Odometer']

plt.figure()
plt.scatter(odometer,car_price,label = "Observed Points", color = "red")

#plt.savefig('car_price.pdf', format='pdf', dpi=1000)

y_data = np.reshape(car_price, (len(car_price), 1))
x_data = np.reshape(odometer, (len(odometer), 1))

# Create the linear regression object
lr_obj = LinearRegression()

# Estiamte coefficients
lr_obj.fit(x_data, y_data)

# Print intercept and coefficients
print("\nThe estimated model parameters are")
print(lr_obj.intercept_)    # This is the intercept \beta_0 in our notation
print(lr_obj.coef_)         # This is \beta_1 in our notation

# estimate using the normal equation
# sklearn handle the intercept automatically. If we need to implement the closed
# form solution we shall add the special feature of all 1s
X = np.column_stack((np.ones(len(car_price)), x_data))
# Convert X to a matrix
X = np.asmatrix(X)

# Estimate linear regression coefficients using normal equation
lin_beta = np.linalg.inv(X.T*X) * X.T * y_data

# plot the fitted linear regression line
x_temp = np.reshape(np.linspace(np.min(x_data), np.max(x_data), 50), (50,1))
y_temp = lr_obj.predict(x_temp)

plt.figure()

plt.plot(x_temp,y_temp)
plt.scatter(odometer,car_price,label = "Observed Points", color = "red")

# fitted a 2nd order polynomial regression line
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(4) 
poly_4 = poly.fit_transform(np.reshape(x_data, (100,1)))
poly_4 = np.asmatrix(poly_4)
lr_obj_4 = LinearRegression()
lr_obj_4.fit(poly_4, y_data)
print(lr_obj_4.intercept_)    # This is the intercept \beta_0 in our notation
print(lr_obj_4.coef_)         # They are \beta_1, \beta_2, \beta_3,\beta_4 in our notation


# plot the fitted 4th order polynomial regression line
x_temp = np.reshape(np.linspace(np.min(x_data), np.max(x_data), 50), (50,1))
poly_temp0_4 = poly.fit_transform(np.reshape(x_temp, (50,1)))
y_temp = lr_obj_4.predict(poly_temp0_4)

plt.plot(x_temp,y_temp)
plt.scatter(odometer,car_price,label = "Observed Points", color = "red")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("C:/Users/rosha/OneDrive/Desktop/ML Practical/s18-2/Salary_positions.csv")


X = df[['Position_Level']].values  
y = df['Salary'].values  

poly = PolynomialFeatures(degree=4)

X_poly = poly.fit_transform(X)

poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)

y_pred = poly_regressor.predict(X_poly)

plt.scatter(X, y, color='red')  
plt.plot(X, y_pred, color='blue')  
plt.title("Polynomial Linear Regression for Salary Prediction")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

level_11 = poly.transform([[11]])  
level_12 = poly.transform([[12]])  

predicted_salary_11 = poly_regressor.predict(level_11)
predicted_salary_12 = poly_regressor.predict(level_12)

print(f"Predicted salary for Level 11: ${predicted_salary_11[0]:,.2f}")
print(f"Predicted salary for Level 12: ${predicted_salary_12[0]:,.2f}")

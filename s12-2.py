import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


data = pd.read_csv("C:/Users/rosha/OneDrive/Desktop/ML Practical/s12-2/Salary_positions.csv")

X = data[['Position_Level']].values  
y = data['Salary'].values   

linear_model = LinearRegression()
linear_model.fit(X, y)

poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_pred_linear = linear_model.predict(X)
mse_linear = mean_squared_error(y, y_pred_linear)

y_pred_poly = poly_model.predict(X_poly)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"Linear Regression MSE: {mse_linear:.2f}")
print(f"Polynomial Regression (Degree 4) MSE: {mse_poly:.2f}")

plt.scatter(X, y, color="blue", label="Actual Salary Data")
plt.plot(X, y_pred_linear, color="red", label="Simple Linear Regression")
plt.plot(X, y_pred_poly, color="green", label="Polynomial Regression (Degree 4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

level_11 = np.array([[11]])
level_12 = np.array([[12]])


salary_11_linear = linear_model.predict(level_11)[0]
salary_12_linear = linear_model.predict(level_12)[0]

salary_11_poly = poly_model.predict(poly_features.transform(level_11))[0]
salary_12_poly = poly_model.predict(poly_features.transform(level_12))[0]

print(f"\nPredicted Salary for Level 11 (Linear): {salary_11_linear:.2f}")
print(f"Predicted Salary for Level 11 (Polynomial): {salary_11_poly:.2f}")

print(f"Predicted Salary for Level 12 (Linear): {salary_12_linear:.2f}")
print(f"Predicted Salary for Level 12 (Polynomial): {salary_12_poly:.2f}")

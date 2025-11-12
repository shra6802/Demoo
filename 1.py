import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('C:/Users/rosha/OneDrive/Desktop/ML Practical/Prac 1/HousingPrediction.csv')
print(df.columns.tolist())
print("First few rows of the dataset:")
print(df.head())
x= df[['Bedrooms','Bathrooms','Sqft_living','Sqft_lot','Floors','Waterfront','View','Condition','Sqft_above','Sqft_basement','Sqft_basement','Yr_built','Yr_renovated']]
y=df ['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred =model.predict(x_test)
mse= mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluated")
print(f'Mean Squared Error(MSE):{mse}')
print(f'R-Squared (R2):{r2}')

comparison_df = pd.DataFrame({'Acutal Price':y_test,'Predicted Price':y_pred})
print("\n Acutal vs Predicted Prices:")
print(comparison_df.head())

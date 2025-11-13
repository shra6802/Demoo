import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/rosha/OneDrive/Desktop/ML Practical/s7-1/Salary_positions.csv")

print(df.isnull().sum())

df = df.dropna()  

X = df[['Position_Level']]  
y = df['Salary']           


model = LinearRegression()

model.fit(X, y)

levels_to_predict = pd.DataFrame([[11], [12]], columns=['Position_Level'])  
predicted_salaries = model.predict(levels_to_predict)

for level, salary in zip([11, 12], predicted_salaries):
    print(f"Predicted salary for Level {level}: ${salary:.2f}")

plt.scatter(X, y, color='blue')  
plt.plot(X, model.predict(X), color='red')  
plt.title('Salary vs Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

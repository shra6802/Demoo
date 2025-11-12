import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

df = pd.read_csv(r'C:/Users/rosha/OneDrive/Desktop/ML Practical/Prac 2/crash.csv')
print(df.head())
df.dropna(inplace=True)
x = df[['Age','Speed']]
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred)
print(f'------------')
print(f'Accuracy:{accuracy:2f}')
print(f'------------')
print('Confusion Matrix')
print(conf_matrix)
print(f'--------------')
print('Classification Report')
print(class_report)

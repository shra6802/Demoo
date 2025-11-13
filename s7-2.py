import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/rosha/OneDrive/Desktop/ML Practical/s7-2/weather.csv")

label_encoder = LabelEncoder()

df['Outlook'] = label_encoder.fit_transform(df['Outlook'])
df['Temperature'] = label_encoder.fit_transform(df['Temperature'])
df['Humidity'] = label_encoder.fit_transform(df['Humidity'])
df['Wind'] = label_encoder.fit_transform(df['Wind'])
df['PlayTennis'] = label_encoder.fit_transform(df['PlayTennis'])

X = df.drop('PlayTennis', axis=1)  
y = df['PlayTennis']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

y_pred = naive_bayes_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy of Naive Bayes Model: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set()

data = pd.read_csv("C:/Users/rosha/OneDrive/Desktop/ML Practical/s26-2/diabetes.csv")
print(data.head())

data.info()

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)
data['Insulin'].fillna(data['Insulin'].median(), inplace=True)
data['BMI'].fillna(data['BMI'].median(), inplace=True)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.33, random_state=42,
                                                    stratify=y)


test_scores = []
train_scores = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(12, 5))
plt.plot(range(1, 21), train_scores, marker='*', label='Train Score')
plt.plot(range(1, 21), test_scores, marker='o', label='Test Score')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Train and Test Scores for different k values')
plt.legend()
plt.show()

best_k = test_scores.index(max(test_scores)) + 1
print(f'Optimal value of K: {best_k}')


knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)


y_pred = knn_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'KNN model accuracy on test data: {accuracy*100:.2f}%')

cnf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

def predict_diabetes(patient_data):
    if len(patient_data) != X.shape[1]:
        raise ValueError(f'Expected {X.shape[1]} features, but got {len(patient_data)} features.')
    

    patient_data_df = pd.DataFrame([patient_data], columns=X.columns)
    

    patient_data_scaled = scaler.transform(patient_data_df)
    

    prediction = knn_best.predict(patient_data_scaled)
    
    return 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'


new_patient_data = [120, 70, 30, 200, 30.0, 25, 1, 0]  
print(f'Prediction for the new patient: {predict_diabetes(new_patient_data)}')

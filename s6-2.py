import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('C:/Users/rosha/OneDrive/Desktop/ML Practical/s6-2/employees_with_clusters.csv')

print("Checking for missing values in the dataset:")
print(df.isnull().sum())  


df_cleaned = df.dropna() 
print("\nData after removing rows with missing values:")
print(df_cleaned.isnull().sum()) 


X = df_cleaned[['Income', 'Age']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X_scaled)

df_cleaned['Cluster'] = kmeans.labels_

print("\nClustered data with labels:")
print(df_cleaned.head())

df_cleaned.to_csv('employees_with_clusters.csv', index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(df_cleaned['Income'], df_cleaned['Age'], c=df_cleaned['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Employees')
plt.xlabel('Income')
plt.ylabel('Age')
plt.colorbar(label='Cluster')
plt.show()

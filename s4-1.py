import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/rosha/OneDrive/Desktop/ML Practical/s4-1/mall_customers.csv')
print("\nFirst few rows of the dataset:")
print(df.head())
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClusters assigned to each customer:")
print(df[['CustomerID', 'Cluster']].head()) 
plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()
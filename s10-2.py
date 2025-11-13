import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

iris = load_iris()
X = iris.data 
y = iris.target 

label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y) 

df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y_numeric  

plt.figure(figsize=(10, 6))


plt.subplot(1, 2, 1)  
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['Species'], cmap='viridis')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.subplot(1, 2, 2) 
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['Species'], cmap='viridis')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# Show the plot
plt.tight_layout()
plt.show()

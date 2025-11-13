import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', np.nan],
    'Age': [24, np.nan, 22, 32, 29, 27],
    'City': ['New York', 'Los Angeles', 'Chicago', np.nan, 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, np.nan, 54000, 62000, 67000]
}

df = pd.DataFrame(data)
print("Original Dataset with Null Values:")
print(df)

print("\nNull Values in Each Column:")
print(df.isnull().sum())

df_cleaned = df.dropna()
print("\nDataset after Removing Rows with Null Values:")
print(df_cleaned)

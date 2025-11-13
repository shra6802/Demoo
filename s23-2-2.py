# This file uses the pre-made 1your_dataset.csv file
# and it displays the cleaned_dataset.csv file after identifying NULL Values.

import pandas as pd

data = pd.read_csv("1your_dataset.csv")

print("Null values in each column before removing:")
print(data.isnull().sum())


data_cleaned = data.dropna()

print("\nNull values in each column after removing:")
print(data_cleaned.isnull().sum())

print(f"\nNumber of rows before removing nulls: {len(data)}")
print(f"Number of rows after removing nulls: {len(data_cleaned)}")

data_cleaned.to_csv("cleaned_dataset.csv", index=True)

import pandas as pd

# Load the CSV file
df = pd.read_csv('Master_Loan_Summary.csv')

# Show first 5 rows
print(df.head())

# Show column names and data types
print(df.info())

# Check missing values
print(df.isnull().sum())

# Basic statistics
print(df.describe())

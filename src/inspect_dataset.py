import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

df = pd.read_csv(DATA_PATH, nrows=1000)

print("Rows loaded:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nInbound distribution:")
print(df["inbound"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
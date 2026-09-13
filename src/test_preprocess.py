import pandas as pd

from preprocess import preprocess_dataframe


DATA_PATH = "data/raw/twcs.csv"


df = pd.read_csv(DATA_PATH, nrows=1000)

processed_df = preprocess_dataframe(df)

print("Rows:", len(processed_df))
print("\nData types:")
print(processed_df.dtypes)

print("\nSample text:")
print(processed_df["text"].head())

print("\nMissing created_at:")
print(processed_df["created_at"].isna().sum())
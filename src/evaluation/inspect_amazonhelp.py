import pandas as pd

DATA_PATH = "data/processed/amazonhelp_english_pairs.csv"

df = pd.read_csv(DATA_PATH)

print("English pairs:", len(df))

sample = df.sample(n=300, random_state=42)

print("\nRandom 300 English customer messages:\n")

for i, text in enumerate(sample["text_customer"], start=1):
    print(f"{i}. {text}")
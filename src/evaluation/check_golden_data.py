import pandas as pd


DATA_PATH = "data/processed/amazonhelp_english_pairs.csv"
OUTPUT_PATH = "eval/golden_set.csv"

df = pd.read_csv(DATA_PATH)

# One evaluation example per customer tweet
df = df.drop_duplicates(subset="tweet_id_customer")

# Random sample
golden = df.sample(n=200, random_state=42).copy()

golden = golden[
    ["tweet_id_customer", "text_customer"]
]

golden.insert(0, "case_id", range(1, len(golden) + 1))
golden["intent_label"] = ""
golden["notes"] = ""

golden.to_csv(OUTPUT_PATH, index=False)

print("Unique customer tweets:", len(df))
print("Golden set created:", len(golden))
print("Saved to:", OUTPUT_PATH)
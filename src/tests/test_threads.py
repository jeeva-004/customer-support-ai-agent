import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from preprocess import preprocess_dataframe
from build_threads import build_threads


DATA_PATH = "data/raw/twcs.csv"


df = pd.read_csv(DATA_PATH, nrows=1000)

df = preprocess_dataframe(df)
df = build_threads(df)

print("Rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nThread sample:")
print(
    df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "in_response_to_tweet_id",
            "parent_text",
        ]
    ].head(10).to_string()
)

print("\nTweets with parent text:")
print(df["parent_text"].notna().sum())
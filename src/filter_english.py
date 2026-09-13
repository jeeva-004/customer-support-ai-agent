import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from concurrent.futures import ProcessPoolExecutor

DATA_PATH = "data/processed/amazonhelp_pairs.csv"
OUTPUT_PATH = "data/processed/amazonhelp_english_pairs.csv"

DetectorFactory.seed = 42


def is_english(text):
    if pd.isna(text):
        return False

    text = str(text).strip()

    if len(text) < 5:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("Total pairs:", len(df))

    texts = df["text_customer"].tolist()

    with ProcessPoolExecutor(max_workers=2) as executor:
        english_flags = list(
            executor.map(is_english, texts, chunksize=100)
        )

    df["is_english"] = english_flags

    english_df = df[df["is_english"]].copy()
    english_df.drop(columns=["is_english"], inplace=True)

    print("English:", len(english_df))
    print("Non-English:", len(df) - len(english_df))

    english_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to: {OUTPUT_PATH}")
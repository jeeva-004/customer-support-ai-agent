import pandas as pd


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if pd.isna(text):
        return ""

    text = str(text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic preprocessing to the dataset."""
    df = df.copy()

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Convert timestamp
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Ensure boolean type
    df["inbound"] = df["inbound"].astype(bool)

    return df
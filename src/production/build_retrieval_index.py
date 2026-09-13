import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/amazonhelp_english_pairs.csv"
GOLDEN_PATH = "eval/golden_set.csv"

EMBEDDINGS_PATH = "data/processed/retrieval_embeddings.npy"
METADATA_PATH = "data/processed/retrieval_metadata.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    golden_ids = set(
        golden["tweet_id_customer"].astype(str)
    )

    df["tweet_id_customer"] = df["tweet_id_customer"].astype(str)

    df = df[
        ~df["tweet_id_customer"].isin(golden_ids)
    ].copy()

    print("Retrieval corpus:", len(df))

    df["text_brand"] = df["text_brand"].fillna("")

    cases = (
        df.groupby(
            ["tweet_id_customer", "text_customer"],
            as_index=False
        )
        .agg({
            "text_brand": lambda replies: "\n".join(
                reply for reply in replies if reply.strip()
            )
        })
    )

    print("Retrieval cases:", len(cases))

    model = SentenceTransformer(MODEL_NAME)

    texts = cases["text_customer"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Embeddings shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)

    np.save(EMBEDDINGS_PATH, embeddings)

    cases.to_csv(
        METADATA_PATH,
        index=False
    )

    print("Full retrieval index saved.")
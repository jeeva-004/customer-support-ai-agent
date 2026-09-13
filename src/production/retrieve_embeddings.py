import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

PAIRS_PATH = "data/processed/amazonhelp_english_pairs.csv"
EMBEDDINGS_PATH = "data/processed/amazonhelp_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_cases():
    df = pd.read_csv(PAIRS_PATH)

    return df[
        ["tweet_id_customer", "text_customer", "text_brand"]
    ].dropna().reset_index(drop=True)


CASES = load_cases()

model = SentenceTransformer(MODEL_NAME)


def load_or_build_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading saved embeddings...")
        return np.load(EMBEDDINGS_PATH)

    print(f"Building embeddings for {len(CASES)} cases...")

    embeddings = model.encode(
        CASES["text_customer"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")

    return embeddings


CASE_EMBEDDINGS = load_or_build_embeddings()


def retrieve_similar_cases(query, top_k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    similarities = CASE_EMBEDDINGS @ query_embedding

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "tweet_id_customer": CASES.iloc[index]["tweet_id_customer"],
            "customer_message": CASES.iloc[index]["text_customer"],
            "agent_response": CASES.iloc[index]["text_brand"],
            "similarity": float(similarities[index])
        })

    return results
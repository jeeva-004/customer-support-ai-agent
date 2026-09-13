import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from classify_intent import classify_intent
from generate_reply import generate_reply


EMBEDDINGS_PATH = "data/processed/retrieval_embeddings.npy"
METADATA_PATH = "data/processed/retrieval_metadata.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve(query, model, embeddings, metadata, top_k=3):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = metadata.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    embedding_model = SentenceTransformer(MODEL_NAME)

    customer_message = "My package has not arrived yet."

    print("\nCustomer:", customer_message)

    intent = classify_intent(customer_message)
    print("Intent:", intent)

    results = retrieve(
        customer_message,
        embedding_model,
        embeddings,
        metadata,
        top_k=3
    )

    retrieved_cases = results[
        ["text_customer", "text_brand"]
    ].to_dict("records")

    print("\nRetrieved cases:", len(retrieved_cases))

    reply = generate_reply(
        customer_message,
        retrieved_cases
    )

    print("\nGenerated reply:")
    print(reply)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAIRS_PATH = "data/processed/amazonhelp_english_pairs.csv"


def load_cases():
    df = pd.read_csv(PAIRS_PATH)

    df = df[
        ["tweet_id_customer", "text_customer", "text_brand"]
    ].dropna()

    return df.reset_index(drop=True)


CASES = load_cases()

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)

case_vectors = vectorizer.fit_transform(
    CASES["text_customer"]
)


def retrieve_similar_cases(query, top_k=3):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        case_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "tweet_id_customer": CASES.iloc[index]["tweet_id_customer"],
            "customer_message": CASES.iloc[index]["text_customer"],
            "agent_response": CASES.iloc[index]["text_brand"],
            "similarity": float(similarities[index]),
        })

    return results
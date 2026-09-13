import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REFERENCE_PATH = "eval/golden_reference.csv"

INTENTS = [
    "delivery_status",
    "delivery_delay_or_missing",
    "return_request",
    "refund_pending",
    "wrong_or_damaged_item",
    "order_change_or_cancellation",
    "out_of_scope",
]


def load_reference_data():
    df = pd.read_csv(REFERENCE_PATH)

    return df[["text_customer", "intent_label"]].dropna()


REFERENCE_DF = load_reference_data()

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)

reference_vectors = vectorizer.fit_transform(
    REFERENCE_DF["text_customer"]
)


def classify_intent(text: str) -> str:
    query_vector = vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        reference_vectors
    )[0]

    best_index = similarities.argmax()

    return REFERENCE_DF.iloc[best_index]["intent_label"]
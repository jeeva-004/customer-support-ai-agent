import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from classify_intent import classify_intent
from retrieve_embeddings import retrieve_similar_cases
from escalate import decide_escalation


queries = [
    "Where is my refund? I still haven't received it.",
    "I received the wrong item and want to return it.",
    "I want to cancel my order.",
    "My Alexa is not working."
]


for query in queries:
    intent = classify_intent(query)
    retrieved_cases = retrieve_similar_cases(query, top_k=3)
    decision = decide_escalation(intent, retrieved_cases)

    print("\n" + "=" * 60)
    print("CUSTOMER:", query)
    print("INTENT:", intent)
    print("DECISION:", decision["decision"])
    print("REASON:", decision["reason"])
    print("TOP SIMILARITY:", retrieved_cases[0]["similarity"])
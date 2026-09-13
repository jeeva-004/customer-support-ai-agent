import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from retrieve_embeddings import retrieve_similar_cases


queries = [
    "Where is my refund? I still haven't received it.",
    "I received the wrong item and want to return it.",
    "I want to cancel my order."
]


for query in queries:

    print("\n" + "=" * 60)
    print("QUERY:", query)

    results = retrieve_similar_cases(query, top_k=3)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Similarity:", result["similarity"])
        print("Customer:", result["customer_message"])
        print("Agent:", result["agent_response"])
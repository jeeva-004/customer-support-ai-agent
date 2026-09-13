import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from generate_reply import generate_reply


queries = [
    "Where is my refund? I still haven't received it.",
    "I received the wrong item and want to return it.",
    "I want to cancel my order."
]


for query in queries:
    print("\n" + "=" * 60)
    print("CUSTOMER:", query)

    result = generate_reply(query)

    print("\nREPLY:")
    print(result["reply"])

    print("\nRETRIEVED CASES:")
    for case in result["retrieved_cases"]:
        print("-", case["customer_message"])
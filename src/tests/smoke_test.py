import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from pipeline import process_customer_message


TEST_MESSAGES = [
    "Where is my package?",
    "My order has not arrived yet.",
    "I want to return this item.",
    "Where is my refund?",
    "I received a damaged product.",
    "I received the wrong item.",
    "I want to cancel my order.",
    "Can I change my order?",
    "My package is still not here.",
    "When will my order arrive?",
    "My refund is still pending.",
    "I need to return something I bought.",
    "The item I received is broken.",
    "I need to cancel an order.",
    "Alexa is not working.",
    "I cannot login to my account.",
    "How do I become an Amazon seller?",
    "Thank you for your help.",
    "My Prime membership has a problem.",
    "My order was supposed to arrive yesterday."
]


for i, message in enumerate(TEST_MESSAGES, start=1):
    result = process_customer_message(message)

    print("\n" + "=" * 60)
    print(f"TEST {i}")
    print("CUSTOMER:", message)
    print("INTENT:", result["intent"])
    print("DECISION:", result["decision"])
    print("REPLY:", result["reply"])
    print("REASON:", result["reason"])
from classify_intent import classify_intent


test_messages = [
    "Where is my package?",
    "My order has not arrived yet.",
    "I want to return this item.",
    "My refund has not arrived.",
    "I received a damaged product.",
    "Please cancel my order.",
    "Thank you for your help.",
]


for message in test_messages:
    intent = classify_intent(message)

    print("Message:", message)
    print("Intent:", intent)
    print()
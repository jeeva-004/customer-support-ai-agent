from pipeline import process_customer_message


queries = [
    "Where is my refund? I still haven't received it.",
    "I received the wrong item and want to return it.",
    "I want to cancel my order.",
    "My Alexa is not working."
]


for query in queries:
    result = process_customer_message(query)

    print("\n" + "=" * 60)
    print("CUSTOMER:", result["customer_message"])
    print("INTENT:", result["intent"])
    print("REPLY:", result["reply"])
    print("DECISION:", result["decision"])
    print("REASON:", result["reason"])
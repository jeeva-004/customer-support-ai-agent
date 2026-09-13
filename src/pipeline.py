from classify_intent import classify_intent
from retrieve_embeddings import retrieve_similar_cases
from generate_reply import generate_reply
from escalate import decide_escalation


def process_customer_message(customer_message):
    intent = classify_intent(customer_message)

    retrieved_cases = retrieve_similar_cases(
        customer_message,
        top_k=3
    )

    escalation = decide_escalation(
        intent,
        retrieved_cases
    )

    reply = None

    if escalation["decision"] == "auto_handle":
        reply = generate_reply(
            customer_message,
            top_k=3
        )["reply"]

    return {
        "customer_message": customer_message,
        "intent": intent,
        "reply": reply,
        "decision": escalation["decision"],
        "reason": escalation["reason"],
        "retrieved_cases": retrieved_cases
    }
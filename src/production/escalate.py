def decide_escalation(intent, retrieved_cases):
    if intent == "out_of_scope":
        return {
            "decision": "escalate",
            "reason": "The request is outside the supported customer support intents."
        }

    if not retrieved_cases:
        return {
            "decision": "escalate",
            "reason": "No relevant historical support case was retrieved."
        }

    top_similarity = retrieved_cases[0]["similarity"]

    if top_similarity < 0.60:
        return {
            "decision": "escalate",
            "reason": "The retrieved historical cases are not sufficiently similar."
        }

    return {
        "decision": "auto_handle",
        "reason": "The request matches a supported intent and has a sufficiently similar historical case."
    }
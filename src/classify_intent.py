import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

REFERENCE_PATH = "eval/intent_reference_examples.csv"

INTENTS = [
    "delivery_status",
    "delivery_delay_or_missing",
    "return_request",
    "refund_pending",
    "wrong_or_damaged_item",
    "order_change_or_cancellation",
    "out_of_scope",
]


def load_reference_examples():
    df = pd.read_csv(REFERENCE_PATH)

    examples = []

    for _, row in df.iterrows():
        examples.append(
            f"Message: {row['text_customer']}\n"
            f"Category: {row['intent_label']}"
        )

    return "\n\n".join(examples)


REFERENCE_EXAMPLES = load_reference_examples()


def build_classification_prompt(text):
    return f"""You are classifying AmazonHelp customer messages.

You have 7 possible categories:

1. delivery_status
   Questions asking where an order/package currently is, tracking status,
   expected delivery date, or normal delivery progress.

2. delivery_delay_or_missing
   The customer says an expected delivery is late, overdue, missing,
   or has not arrived when it should have.

3. return_request
   The customer wants to return an item or asks how to return an item.

4. refund_pending
   The customer is specifically waiting for a refund or asking about
   a refund that has not been received.

5. wrong_or_damaged_item
   The customer received an item that is damaged, defective, broken,
   incorrect, or different from what they ordered.

6. order_change_or_cancellation
   The customer explicitly wants to change or cancel an order.

7. out_of_scope
   Anything that is NOT one of the six supported intents above.

IMPORTANT:
- Decide whether the message belongs to a supported intent BEFORE choosing
  one of the six support categories.
- If it does not clearly match one of the six supported intents, choose
  out_of_scope.
- Do NOT force a message into a support category merely because it contains
  words such as "order", "delivery", "refund", or "Amazon".
- Prime membership, Alexa/Echo, Kindle, Fire TV, Amazon Music, account/login/
  security, Amazon Pay, payments/cards, promotions/pricing, seller issues,
  general product issues, complaints, opinions, thank-you messages,
  status updates, and unrelated/general conversation are out_of_scope.
- A message mentioning an order is NOT automatically an order-change request.
  It must explicitly ask to change or cancel an order.
- A message mentioning delivery is NOT automatically a delivery intent.
  It must actually ask about delivery status or report a delayed/missing
  delivery.
- A message mentioning money/payment is NOT automatically refund_pending.
  It must specifically concern a pending refund.
- If the message is only a complaint, acknowledgement, thank-you, or update
  without one of the six supported requests, choose out_of_scope.

Use the following labelled examples as additional guidance:

{REFERENCE_EXAMPLES}

Now classify this message:

Message: {text}

Return ONLY one category name.
"""

def validate_intent(intent: str) -> str:
    intent = intent.strip()

    if intent not in INTENTS:
        return "out_of_scope"

    return intent

def rule_based_intent(text):
    text_lower = text.lower()

    # Strong out-of-scope signals
    oos_terms = [
        "prime membership",
        "prime trial",
        "alexa",
        "echo",
        "kindle",
        "fire tv",
        "amazon music",
        "amazon pay",
        "seller account",
        "seller",
    ]

    if any(term in text_lower for term in oos_terms):
        return "out_of_scope"

    # Strong return signals
    if "return" in text_lower:
        return "return_request"

    # Strong refund signals
    if "refund" in text_lower:
        return "refund_pending"

    # Strong wrong/damaged item signals
    wrong_item_terms = [
        "wrong item",
        "wrong product",
        "damaged item",
        "damaged product",
        "broken item",
        "defective item",
        "incorrect item",
    ]

    if any(term in text_lower for term in wrong_item_terms):
        return "wrong_or_damaged_item"

    # Strong cancellation signals
    cancel_terms = [
        "cancel my order",
        "cancel the order",
        "want to cancel my order",
        "want to cancel the order",
    ]

    if any(term in text_lower for term in cancel_terms):
        return "order_change_or_cancellation"

    return None


def classify_intent(text: str) -> str:

    # First: deterministic classification for obvious cases
    rule_result = rule_based_intent(text)

    if rule_result is not None:
        return rule_result

    # Otherwise: use LLM
    prompt = build_classification_prompt(text)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            },
        },
        timeout=600,
    )

    response.raise_for_status()

    result = response.json()["response"]

    return validate_intent(result)
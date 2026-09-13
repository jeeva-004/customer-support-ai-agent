import os
import sys
import json
import requests
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from generate_reply import generate_reply

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

INPUT_PATH = "eval/golden_test.csv"
OUTPUT_PATH = "eval/evidence_judge_results.csv"


def judge_evidence(customer_message, reply, retrieved_cases):
    evidence = "\n\n".join(
        f"Historical AmazonHelp response {i}:\n{case['agent_response']}"
        for i, case in enumerate(retrieved_cases, 1)
    )

    prompt = f"""You are evaluating whether an AI customer support reply
is grounded in retrieved historical AmazonHelp responses.

Customer message:
{customer_message}

AI-generated reply:
{reply}

Retrieved historical responses:
{evidence}

Identify the factual or actionable claims made by the AI reply.

For EACH claim, classify it as exactly one of:

supported
unsupported
uncertain

Rules:
- Greetings, apologies, and polite phrases are not claims.
- A claim is supported only when the historical responses provide evidence for it.
- Do not assume information that is not explicitly present.
- Links, policies, timelines, actions, and instructions count as claims.
- If a claim is not supported by the evidence, mark it unsupported.

Return ONLY valid JSON in this format:

{{
  "claims": [
    {{
      "claim": "short description of the claim",
      "label": "supported"
    }}
  ]
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            },
        },
        timeout=600,
    )

    response.raise_for_status()

    return json.loads(response.json()["response"])


def main():
    df = pd.read_csv(INPUT_PATH)

    # First run: 10 supported-intent cases.
    sample = df[df["intent_label"] != "out_of_scope"].head(10)

    results = []

    for _, row in sample.iterrows():
        customer_message = row["text_customer"]

        generated = generate_reply(
            customer_message,
            top_k=3
        )

        reply = generated["reply"]
        retrieved_cases = generated["retrieved_cases"]

        judgment = judge_evidence(
            customer_message,
            reply,
            retrieved_cases
        )

        claims = judgment.get("claims", [])

        supported = sum(
            1 for claim in claims
            if claim.get("label") == "supported"
        )

        unsupported = sum(
            1 for claim in claims
            if claim.get("label") == "unsupported"
        )

        uncertain = sum(
            1 for claim in claims
            if claim.get("label") == "uncertain"
        )

        total = supported + unsupported + uncertain

        support_rate = (
            supported / total
            if total > 0
            else None
        )

        results.append({
            "case_id": row["case_id"],
            "customer_message": customer_message,
            "reply": reply,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "uncertain_claims": uncertain,
            "evidence_support_rate": support_rate,
            "claims": json.dumps(claims)
        })

        print("\n" + "=" * 60)
        print("CASE:", row["case_id"])
        print("CUSTOMER:", customer_message)
        print("REPLY:", reply)
        print("CLAIMS:")

        for claim in claims:
            print(
                f"- [{claim.get('label')}] "
                f"{claim.get('claim')}"
            )

        print(
            "SUPPORT RATE:",
            f"{support_rate:.2%}" if support_rate is not None else "N/A"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    valid_rates = results_df[
        results_df["evidence_support_rate"].notna()
    ]["evidence_support_rate"]

    print("\n" + "=" * 60)
    print("EVIDENCE EVALUATION")
    print("=" * 60)

    if len(valid_rates) > 0:
        print(
            "Average evidence support rate:",
            f"{valid_rates.mean():.2%}"
        )
    else:
        print("Average evidence support rate: N/A")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
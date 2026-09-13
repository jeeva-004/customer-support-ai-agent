import os
import sys
import json
import requests
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from retrieve_embeddings import retrieve_similar_cases
from generate_reply import generate_reply


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

INPUT_PATH = "eval/golden_test.csv"
OUTPUT_PATH = "eval/reply_judge_results.csv"


def judge_reply(customer_message, reply, retrieved_cases):
    evidence = "\n\n".join(
        f"Historical AmazonHelp response:\n{case['agent_response']}"
        for case in retrieved_cases
    )

    prompt = f"""You are evaluating an Amazon customer support AI reply.

Customer message:
{customer_message}

AI-generated reply:
{reply}

Historical AmazonHelp responses used as evidence:
{evidence}

Score the reply from 1 to 5 for each criterion:

1. relevance:
Does the reply address the customer's actual issue?

2. helpfulness:
Does the reply provide useful support without unnecessary content?

3. grounding:
Is the reply supported by the historical AmazonHelp responses?
Penalize invented policies, actions, links, timelines, or details.

4. overall:
Overall quality of the reply.

Return ONLY valid JSON in this exact format:

{{
  "relevance": 1,
  "helpfulness": 1,
  "grounding": 1,
  "overall": 1
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

    # Evaluate 10 cases first.
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

        scores = judge_reply(
            customer_message,
            reply,
            retrieved_cases
        )

        results.append({
            "case_id": row["case_id"],
            "customer_message": customer_message,
            "reply": reply,
            **scores
        })

        print("\n" + "=" * 60)
        print("CASE:", row["case_id"])
        print("CUSTOMER:", customer_message)
        print("REPLY:", reply)
        print("SCORES:", scores)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("AVERAGES")
    print("=" * 60)

    print("Relevance:", results_df["relevance"].mean())
    print("Helpfulness:", results_df["helpfulness"].mean())
    print("Grounding:", results_df["grounding"].mean())
    print("Overall:", results_df["overall"].mean())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
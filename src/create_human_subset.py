import pandas as pd

from generate_reply import generate_reply

INPUT_PATH = "eval/golden_test.csv"
OUTPUT_PATH = "eval/human_agreement.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    sample = df[df["intent_label"] != "out_of_scope"].head(10)

    results = []

    for _, row in sample.iterrows():
        generated = generate_reply(
            row["text_customer"],
            top_k=3
        )

        results.append({
            "case_id": row["case_id"],
            "customer_message": row["text_customer"],
            "generated_reply": generated["reply"],
            "human_relevant": "",
            "human_helpful": "",
            "human_grounded": ""
        })

    pd.DataFrame(results).to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Created: {OUTPUT_PATH}")
    print("Open the CSV and manually fill:")
    print("human_relevant  -> 1 = yes, 0 = no")
    print("human_helpful   -> 1 = yes, 0 = no")
    print("human_grounded  -> 1 = yes, 0 = no")


if __name__ == "__main__":
    main()
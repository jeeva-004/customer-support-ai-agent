import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from classify_intent import classify_intent


TEST_PATH = "eval/golden_test.csv"
OUTPUT_PATH = "eval/intent_predictions.csv"


def main():
    df = pd.read_csv(TEST_PATH)

    results = []

    for _, row in df.iterrows():
        actual = row["intent_label"]
        predicted = classify_intent(row["text_customer"])

        results.append({
            "case_id": row["case_id"],
            "text_customer": row["text_customer"],
            "actual": actual,
            "predicted": predicted
        })

    results_df = pd.DataFrame(results)

    accuracy = accuracy_score(
        results_df["actual"],
        results_df["predicted"]
    )

    print("=" * 60)
    print("AUTOMATED INTENT EVALUATION")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            results_df["actual"],
            results_df["predicted"],
            zero_division=0
        )
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
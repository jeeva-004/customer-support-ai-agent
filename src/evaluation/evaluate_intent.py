import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from classify_intent import classify_intent

TEST_PATH = "eval/golden_test.csv"
OUTPUT_PATH = "eval/intent_predictions.csv"

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
results_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(classification_report(
    results_df["actual"],
    results_df["predicted"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    results_df["actual"],
    results_df["predicted"]
))

print(f"\nSaved predictions to: {OUTPUT_PATH}")
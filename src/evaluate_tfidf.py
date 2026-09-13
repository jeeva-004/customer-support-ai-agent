import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from classify_tfidf import classify_intent


TEST_PATH = "eval/golden_test.csv"


df = pd.read_csv(TEST_PATH)

predictions = []

for _, row in df.iterrows():
    predicted = classify_intent(row["text_customer"])
    predictions.append(predicted)


df["predicted"] = predictions

print("=" * 60)
print("TF-IDF Classification Report")
print("=" * 60)

print(
    classification_report(
        df["intent_label"],
        df["predicted"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(df["intent_label"], df["predicted"]))

accuracy = (df["intent_label"] == df["predicted"]).mean()

print(f"\nAccuracy: {accuracy:.2%}")
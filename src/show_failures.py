import pandas as pd

df = pd.read_csv("eval/intent_predictions.csv")

wrong = df[df["actual"] != df["predicted"]]

print("=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

print(
    wrong[
        ["case_id", "text_customer", "actual", "predicted"]
    ].to_string(index=False)
)

print("\nTotal wrong:", len(wrong))
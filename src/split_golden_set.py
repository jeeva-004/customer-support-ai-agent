import pandas as pd

INPUT_PATH = "eval/golden_set.csv"

REFERENCE_PATH = "eval/golden_reference.csv"
TEST_PATH = "eval/golden_test.csv"

df = pd.read_csv(INPUT_PATH)

# Keep the split fixed and reproducible
reference = df.sample(n=150, random_state=42)
test = df.drop(reference.index)

reference = reference.sort_values("case_id")
test = test.sort_values("case_id")

reference.to_csv(REFERENCE_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print("Total:", len(df))
print("Reference:", len(reference))
print("Held-out test:", len(test))

print("\nReference label distribution:")
print(reference["intent_label"].value_counts())

print("\nTest label distribution:")
print(test["intent_label"].value_counts())
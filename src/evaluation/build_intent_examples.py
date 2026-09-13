import pandas as pd

INPUT_PATH = "eval/golden_reference.csv"
OUTPUT_PATH = "eval/intent_reference_examples.csv"

df = pd.read_csv(INPUT_PATH)

examples = (
    df.groupby("intent_label", group_keys=False)
      .apply(lambda group: group.sample(
          n=min(5, len(group)),
          random_state=42
      ))
      .reset_index(drop=True)
)

examples = examples[
    ["case_id", "text_customer", "intent_label"]
]

examples.to_csv(OUTPUT_PATH, index=False)

print("Total reference examples:", len(examples))
print("\nExamples per intent:")
print(examples["intent_label"].value_counts())

print("\nSelected examples:")
print(examples.to_string(index=False))
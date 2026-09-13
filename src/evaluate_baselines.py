import pandas as pd

TEST_PATH = "eval/golden_test.csv"


if __name__ == "__main__":
    df = pd.read_csv(TEST_PATH)

    majority_label = df["intent_label"].value_counts().idxmax()

    correct = (
        df["intent_label"] == majority_label
    ).sum()

    accuracy = correct / len(df)

    print("Majority-class baseline")
    print("-----------------------")
    print("Majority label:", majority_label)
    print(f"Correct: {correct}/{len(df)}")
    print(f"Accuracy: {accuracy:.2%}")
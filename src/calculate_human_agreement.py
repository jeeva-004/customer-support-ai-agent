import pandas as pd

INPUT_PATH = "eval/human_agreement.csv"


def agreement(column):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return values.mean()


def main():
    global df

    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("HUMAN REVIEW SUBSET")
    print("=" * 60)

    for column, name in [
        ("human_relevant", "Relevance"),
        ("human_helpful", "Helpfulness"),
        ("human_grounded", "Grounding"),
    ]:
        score = agreement(column)

        print(f"{name}: {score:.2%}")

    print("\nCases reviewed:", len(df))


if __name__ == "__main__":
    main()
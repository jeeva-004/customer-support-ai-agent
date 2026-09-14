import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "production")))

from pipeline import process_customer_message


result = process_customer_message("i think this item not for me")
print("Intent:", result["intent"])
print("Decision:", result["decision"])
print("Reason:", result["reason"])
print("Reply:", result["reply"])


# INPUT_PATH = "eval/human_agreement.csv"


# def agreement(column):
#     values = pd.to_numeric(df[column], errors="coerce").dropna()
#     return values.mean()


# def main():
#     global df

#     df = pd.read_csv(INPUT_PATH)

#     print("=" * 60)
#     print("HUMAN REVIEW SUBSET")
#     print("=" * 60)

#     for column, name in [
#         ("human_relevant", "Relevance"),
#         ("human_helpful", "Helpfulness"),
#         ("human_grounded", "Grounding"),
#     ]:
#         score = agreement(column)

#         print(f"{name}: {score:.2%}")

#     print("\nCases reviewed:", len(df))


# if __name__ == "__main__":
#     main()
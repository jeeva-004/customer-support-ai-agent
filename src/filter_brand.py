import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
BRAND = "AmazonHelp"
OUTPUT_PATH = "data/processed/amazonhelp_pairs.csv"

def load_brand_replies():
    brand_replies = []

    for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
        rows = chunk[
            (chunk["author_id"] == BRAND)
            & (chunk["inbound"] == False)
        ]

        brand_replies.append(rows)

    return pd.concat(brand_replies, ignore_index=True)


def load_customer_tweets(tweet_ids):
    customer_tweets = []

    for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
        rows = chunk[chunk["tweet_id"].isin(tweet_ids)]
        customer_tweets.append(rows)

    return pd.concat(customer_tweets, ignore_index=True)


if __name__ == "__main__":
    brand_replies = load_brand_replies()

    parent_ids = (
        brand_replies["in_response_to_tweet_id"]
        .dropna()
        .astype("int64")
        .unique()
    )

    customer_tweets = load_customer_tweets(parent_ids)

    customer_tweets = customer_tweets[
        customer_tweets["inbound"] == True
    ]

    pairs = brand_replies.merge(
        customer_tweets,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_brand", "_customer")
    )

    print("AmazonHelp brand replies:", len(brand_replies))
    print("Customer tweets found:", len(customer_tweets))
    print("Customer → AmazonHelp pairs:", len(pairs))

    print("\nSample pairs:")

    print(
        pairs[
            [
                "tweet_id_customer",
                "text_customer",
                "tweet_id_brand",
                "text_brand"
            ]
        ].head(10).to_string()
    )

    pairs.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")
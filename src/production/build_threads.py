import pandas as pd


def build_threads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add parent tweet information so each tweet can be
    connected to the tweet it is responding to.
    """

    df = df.copy()

    # Map tweet_id -> text
    tweet_text_map = df.set_index("tweet_id")["text"]

    # Find the parent tweet's text
    df["parent_text"] = df["in_response_to_tweet_id"].map(tweet_text_map)

    return df
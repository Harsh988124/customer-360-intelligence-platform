"""
Sentiment Analysis
Customer 360 Intelligence & Retention Platform

Creates customer-level sentiment features from support tickets.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_database_url():

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not all([host, port, database, username, password]):
        raise ValueError(
            "Missing database environment variables."
        )

    return (
        f"postgresql+psycopg2://{username}:{password}"
        f"@{host}:{port}/{database}"
    )


# ---------------------------------------------------------
# LOAD SUPPORT DATA
# ---------------------------------------------------------

def load_support_data(engine):

    query = """
        SELECT
            ticket_id,
            customer_id,
            ticket_date,
            customer_sentiment
        FROM support_tickets
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df):,} support tickets.")

    return df


# ---------------------------------------------------------
# CLEAN SENTIMENT
# ---------------------------------------------------------

def clean_sentiment(df):

    df["customer_sentiment"] = (
        df["customer_sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only known sentiment categories
    valid_sentiments = [
        "positive",
        "negative",
        "neutral"
    ]

    df = df[
        df["customer_sentiment"].isin(
            valid_sentiments
        )
    ].copy()

    return df


# ---------------------------------------------------------
# CREATE SENTIMENT FEATURES
# ---------------------------------------------------------

def calculate_sentiment_features(df):

    # Total tickets
    total_tickets = (
        df.groupby("customer_id")
        .size()
        .reset_index(name="total_sentiment_tickets")
    )

    # Positive tickets
    positive = (
        df[df["customer_sentiment"] == "positive"]
        .groupby("customer_id")
        .size()
        .reset_index(name="positive_ticket_count")
    )

    # Negative tickets
    negative = (
        df[df["customer_sentiment"] == "negative"]
        .groupby("customer_id")
        .size()
        .reset_index(name="negative_ticket_count")
    )

    # Neutral tickets
    neutral = (
        df[df["customer_sentiment"] == "neutral"]
        .groupby("customer_id")
        .size()
        .reset_index(name="neutral_ticket_count")
    )

    # Combine
    result = total_tickets

    result = result.merge(
        positive,
        on="customer_id",
        how="left"
    )

    result = result.merge(
        negative,
        on="customer_id",
        how="left"
    )

    result = result.merge(
        neutral,
        on="customer_id",
        how="left"
    )

    # Missing sentiment counts = 0
    count_columns = [
        "positive_ticket_count",
        "negative_ticket_count",
        "neutral_ticket_count"
    ]

    result[count_columns] = (
        result[count_columns]
        .fillna(0)
        .astype(int)
    )

    # Sentiment rates
    result["negative_sentiment_rate"] = (
        result["negative_ticket_count"]
        / result["total_sentiment_tickets"]
    )

    result["positive_sentiment_rate"] = (
        result["positive_ticket_count"]
        / result["total_sentiment_tickets"]
    )

    result["neutral_sentiment_rate"] = (
        result["neutral_ticket_count"]
        / result["total_sentiment_tickets"]
    )

    # Simple sentiment score
    #
    # Positive = +1
    # Neutral  =  0
    # Negative = -1
    #
    # Range: -1 to +1

    result["sentiment_score"] = (
        (
            result["positive_ticket_count"]
            - result["negative_ticket_count"]
        )
        / result["total_sentiment_tickets"]
    )

    result["negative_sentiment_rate"] = (
        result["negative_sentiment_rate"].round(4)
    )

    result["positive_sentiment_rate"] = (
        result["positive_sentiment_rate"].round(4)
    )

    result["neutral_sentiment_rate"] = (
        result["neutral_sentiment_rate"].round(4)
    )

    result["sentiment_score"] = (
        result["sentiment_score"].round(4)
    )

    return result


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_sentiment_features(
    result,
    engine
):

    result.to_sql(
        "customer_sentiment",
        engine,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print(
        "\nCustomer sentiment table "
        "saved successfully."
    )


# ---------------------------------------------------------
# DISPLAY SUMMARY
# ---------------------------------------------------------

def show_summary(
    df,
    result
):

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)

    print(
        f"Valid support tickets: "
        f"{len(df):,}"
    )

    print(
        f"Customers with support tickets: "
        f"{len(result):,}"
    )

    print("\nTicket Sentiment Distribution:")

    sentiment_distribution = (
        df["customer_sentiment"]
        .value_counts()
    )

    print(
        sentiment_distribution
    )

    print("\nAverage Sentiment Metrics:")

    print(
        f"Average negative rate: "
        f"{result['negative_sentiment_rate'].mean():.4f}"
    )

    print(
        f"Average positive rate: "
        f"{result['positive_sentiment_rate'].mean():.4f}"
    )

    print(
        f"Average sentiment score: "
        f"{result['sentiment_score'].mean():.4f}"
    )

    print("\nCustomer Sentiment Categories:")

    def classify_sentiment(score):

        if score >= 0.25:
            return "Positive"

        elif score <= -0.25:
            return "Negative"

        else:
            return "Neutral"

    result["sentiment_category"] = (
        result["sentiment_score"]
        .apply(classify_sentiment)
    )

    print(
        result["sentiment_category"]
        .value_counts()
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("CUSTOMER SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)

    database_url = get_database_url()

    engine = create_engine(
        database_url
    )

    # Load
    print("\nLoading support ticket data...")

    df = load_support_data(
        engine
    )

    # Clean
    print("\nCleaning sentiment values...")

    df = clean_sentiment(
        df
    )

    print(
        f"Valid sentiment records: "
        f"{len(df):,}"
    )

    # Calculate
    print(
        "\nCalculating customer-level "
        "sentiment features..."
    )

    result = calculate_sentiment_features(
        df
    )

    # Save
    save_sentiment_features(
        result,
        engine
    )

    # Summary
    show_summary(
        df,
        result
    )

    print("\n" + "=" * 60)
    print(
        "PHASE 11 SENTIMENT ANALYSIS "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
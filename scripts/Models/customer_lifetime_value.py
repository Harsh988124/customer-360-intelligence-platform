"""
Customer Lifetime Value (CLV)
Customer 360 Intelligence & Retention Platform

Calculates estimated customer lifetime value using:
- Monthly subscription price
- Customer tenure
- Transaction behavior
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()


def get_database_url():
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not all([host, port, database, username, password]):
        raise ValueError("Missing database environment variables.")

    return (
        f"postgresql+psycopg2://{username}:{password}"
        f"@{host}:{port}/{database}"
    )


def load_customer_data(engine):
    query = """
        SELECT
            customer_id,
            total_spend,
            total_transactions,
            average_spend_per_transaction,
            monthly_price,
            tenure_months
        FROM customer_360
    """

    return pd.read_sql(query, engine)


def calculate_clv(df):
    # Estimate monthly revenue from historical transaction spend.
    df["estimated_monthly_revenue"] = (
        df["total_spend"] /
        df["tenure_months"].clip(lower=1)
    )

    # Estimate customer lifetime using current tenure.
    # This is a simple portfolio-friendly CLV assumption.
    df["estimated_lifetime_months"] = (
        df["tenure_months"] * 1.5
    ).clip(lower=12)

    df["estimated_clv"] = (
        df["estimated_monthly_revenue"]
        * df["estimated_lifetime_months"]
    )

    # Round financial values.
    financial_columns = [
        "estimated_monthly_revenue",
        "estimated_lifetime_months",
        "estimated_clv",
    ]

    for column in financial_columns:
        df[column] = df[column].round(2)

    return df


def assign_clv_segment(df):
    df["clv_segment"] = pd.qcut(
        df["estimated_clv"],
        q=4,
        labels=[
            "Low CLV",
            "Medium CLV",
            "High CLV",
            "Very High CLV",
        ],
        duplicates="drop",
    )

    return df


def save_clv(df, engine):
    df.to_sql(
        "customer_clv",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
    )

    print("\nCustomer CLV table saved successfully.")


def show_summary(df):
    print("\n" + "=" * 60)
    print("CUSTOMER LIFETIME VALUE SUMMARY")
    print("=" * 60)

    print(f"Customers processed: {len(df):,}")

    print("\nAverage CLV:")
    print(f"${df['estimated_clv'].mean():,.2f}")

    print("\nMinimum CLV:")
    print(f"${df['estimated_clv'].min():,.2f}")

    print("\nMaximum CLV:")
    print(f"${df['estimated_clv'].max():,.2f}")

    print("\nCLV Segment Distribution:")
    print(df["clv_segment"].value_counts().sort_index())


def main():
    print("=" * 60)
    print("CUSTOMER LIFETIME VALUE PIPELINE")
    print("=" * 60)

    database_url = get_database_url()
    engine = create_engine(database_url)

    print("\nLoading Customer 360 data...")
    df = load_customer_data(engine)

    print(f"Loaded {len(df):,} customers.")

    print("\nCalculating CLV...")
    df = calculate_clv(df)

    print("Assigning CLV segments...")
    df = assign_clv_segment(df)

    save_clv(df, engine)

    show_summary(df)

    print("\n" + "=" * 60)
    print("PHASE 8 CLV PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
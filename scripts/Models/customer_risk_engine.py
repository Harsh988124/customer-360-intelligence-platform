"""
Phase 13 - Customer Risk Engine

Purpose:
Combine multiple customer intelligence signals into one
customer-level risk score.

Inputs:
    customer_360
    churn_predictions
    customer_clv
    customer_sentiment
    customer_anomalies

Output:
    customer_risk

Risk signals:
    1. Churn probability
    2. CLV / business impact
    3. Customer sentiment
    4. Anomaly detection
    5. Customer engagement
    6. Support/payment issues
"""

import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine


# ============================================================
# 1. DATABASE CONNECTION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_customer_360():

    print("\nLoading Customer 360 data...")

    query = """
        SELECT
            customer_id,
            total_logins,
            total_session_minutes,
            total_support_tickets,
            payment_failure_rate
        FROM customer_360
    """

    df = pd.read_sql(query, engine)

    print(f"Customer 360 loaded: {len(df):,} customers.")

    return df


def load_churn_predictions():

    print("\nLoading churn predictions...")

    query = """
        SELECT
            customer_id,
            churn_probability_ml,
            predicted_churn,
            model_name
        FROM churn_predictions
    """

    df = pd.read_sql(query, engine)

    print(f"Churn predictions loaded: {len(df):,} customers.")

    return df


def load_clv():

    print("\nLoading CLV data...")

    query = """
        SELECT *
        FROM customer_clv
    """

    df = pd.read_sql(query, engine)

    print(f"CLV data loaded: {len(df):,} customers.")

    return df


def load_sentiment():

    print("\nLoading sentiment data...")

    query = """
        SELECT *
        FROM customer_sentiment
    """

    df = pd.read_sql(query, engine)

    print(f"Sentiment data loaded: {len(df):,} customers.")

    return df


def load_anomalies():

    print("\nLoading anomaly data...")

    query = """
        SELECT *
        FROM customer_anomalies
    """

    df = pd.read_sql(query, engine)

    print(f"Anomaly data loaded: {len(df):,} customers.")

    return df


# ============================================================
# 3. FIND CLV COLUMN
# ============================================================

def get_clv_column(df):

    possible_columns = [
        "estimated_clv",
        "clv",
        "customer_lifetime_value"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "CLV column not found in customer_clv.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 4. FIND SENTIMENT COLUMN
# ============================================================

def get_sentiment_column(df):

    possible_columns = [
        "sentiment_score",
        "average_sentiment_score"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Sentiment score column not found in customer_sentiment.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 5. FIND ANOMALY COLUMNS
# ============================================================

def get_anomaly_columns(df):

    score_columns = [
        "anomaly_score",
        "isolation_score"
    ]

    flag_columns = [
        "anomaly_flag",
        "is_anomaly",
        "final_anomaly"
    ]

    score_column = None
    flag_column = None

    for column in score_columns:

        if column in df.columns:
            score_column = column
            break

    for column in flag_columns:

        if column in df.columns:
            flag_column = column
            break

    if score_column is None:

        raise ValueError(
            "Anomaly score column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if flag_column is None:

        raise ValueError(
            "Anomaly flag column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return score_column, flag_column


# ============================================================
# 6. MIN-MAX SCALING
# ============================================================

def min_max_scale(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            0.0,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    )


# ============================================================
# 7. COMBINE ALL DATA
# ============================================================

def combine_data(
    customer_360,
    churn,
    clv,
    sentiment,
    anomalies
):

    print("\nCombining all customer intelligence layers...")

    # --------------------------------------------------------
    # CLV
    # --------------------------------------------------------

    clv_column = get_clv_column(clv)

    clv_data = clv[
        [
            "customer_id",
            clv_column
        ]
    ].copy()

    clv_data = clv_data.rename(
        columns={
            clv_column: "clv"
        }
    )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    sentiment_column = get_sentiment_column(
        sentiment
    )

    sentiment_data = sentiment[
        [
            "customer_id",
            sentiment_column
        ]
    ].copy()

    sentiment_data = sentiment_data.rename(
        columns={
            sentiment_column: "sentiment_score"
        }
    )

    # --------------------------------------------------------
    # ANOMALIES
    # --------------------------------------------------------

    anomaly_score_column, anomaly_flag_column = (
        get_anomaly_columns(anomalies)
    )

    anomaly_data = anomalies[
        [
            "customer_id",
            anomaly_score_column,
            anomaly_flag_column
        ]
    ].copy()

    anomaly_data = anomaly_data.rename(
        columns={
            anomaly_score_column: "anomaly_score",
            anomaly_flag_column: "anomaly_flag"
        }
    )

    # --------------------------------------------------------
    # CHURN
    #
    # IMPORTANT:
    # Your actual column is churn_probability_ml
    # --------------------------------------------------------

    churn_data = churn[
        [
            "customer_id",
            "churn_probability_ml",
            "predicted_churn",
            "model_name"
        ]
    ].copy()

    churn_data = churn_data.rename(
        columns={
            "churn_probability_ml":
                "churn_probability"
        }
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = customer_360.copy()

    df = df.merge(
        churn_data,
        on="customer_id",
        how="left"
    )

    df = df.merge(
        clv_data,
        on="customer_id",
        how="left"
    )

    df = df.merge(
        sentiment_data,
        on="customer_id",
        how="left"
    )

    df = df.merge(
        anomaly_data,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # CHECK DUPLICATES
    # --------------------------------------------------------

    duplicate_count = df["customer_id"].duplicated().sum()

    if duplicate_count > 0:

        print(
            f"WARNING: {duplicate_count} duplicate "
            "customer records detected."
        )

        df = df.drop_duplicates(
            subset="customer_id",
            keep="first"
        )

    print(
        f"Combined dataset: "
        f"{len(df):,} customers."
    )

    return df


# ============================================================
# 8. CLEAN NUMERIC DATA
# ============================================================

def clean_data(df):

    numeric_columns = [
        "total_logins",
        "total_session_minutes",
        "total_support_tickets",
        "payment_failure_rate",
        "churn_probability",
        "predicted_churn",
        "clv",
        "sentiment_score",
        "anomaly_score",
        "anomaly_flag"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    df["churn_probability"] = (
        df["churn_probability"]
        .fillna(0)
        .clip(0, 1)
    )

    df["predicted_churn"] = (
        df["predicted_churn"]
        .fillna(0)
        .astype(int)
    )

    df["clv"] = (
        df["clv"]
        .fillna(df["clv"].median())
    )

    # Customers without support tickets are treated
    # as having neutral sentiment.
    df["sentiment_score"] = (
        df["sentiment_score"]
        .fillna(0)
        .clip(-1, 1)
    )

    df["anomaly_score"] = (
        df["anomaly_score"]
        .fillna(0)
    )

    df["anomaly_flag"] = (
        df["anomaly_flag"]
        .fillna(0)
        .astype(int)
    )

    df["total_logins"] = (
        df["total_logins"]
        .fillna(0)
    )

    df["total_session_minutes"] = (
        df["total_session_minutes"]
        .fillna(0)
    )

    df["total_support_tickets"] = (
        df["total_support_tickets"]
        .fillna(0)
    )

    df["payment_failure_rate"] = (
        df["payment_failure_rate"]
        .fillna(0)
        .clip(0, 1)
    )

    return df


# ============================================================
# 9. CHURN RISK
# ============================================================

def calculate_churn_risk(df):

    # ML probability is already between 0 and 1.
    # Convert to 0-100.

    df["churn_risk_score"] = (
        df["churn_probability"] * 100
    ).clip(0, 100)

    return df


# ============================================================
# 10. CLV BUSINESS IMPACT
# ============================================================

def calculate_clv_impact(df):

    # High CLV means higher business impact
    # if the customer becomes at risk.

    df["clv_impact_score"] = min_max_scale(
        df["clv"]
    )

    return df


# ============================================================
# 11. SENTIMENT RISK
# ============================================================

def calculate_sentiment_risk(df):

    """
    Convert sentiment:

        +1 = Positive
         0 = Neutral
        -1 = Negative

    Risk:

        +1 -> 0 risk
         0 -> 50 risk
        -1 -> 100 risk
    """

    df["sentiment_risk_score"] = (
        (1 - df["sentiment_score"])
        / 2
        * 100
    ).clip(0, 100)

    return df


# ============================================================
# 12. ANOMALY RISK
# ============================================================

def calculate_anomaly_risk(df):

    anomaly_scaled = min_max_scale(
        df["anomaly_score"]
    )

    # If the customer is explicitly flagged as anomalous,
    # ensure they receive at least moderate anomaly risk.

    df["anomaly_risk_score"] = np.where(
        df["anomaly_flag"] == 1,
        np.maximum(
            anomaly_scaled,
            50
        ),
        anomaly_scaled
    )

    df["anomaly_risk_score"] = (
        df["anomaly_risk_score"]
        .clip(0, 100)
    )

    return df


# ============================================================
# 13. ENGAGEMENT RISK
# ============================================================

def calculate_engagement_risk(df):

    login_score = min_max_scale(
        df["total_logins"]
    )

    session_score = min_max_scale(
        df["total_session_minutes"]
    )

    # Higher engagement = lower risk.

    engagement_score = (
        login_score * 0.50
        +
        session_score * 0.50
    )

    df["engagement_risk_score"] = (
        100 - engagement_score
    ).clip(0, 100)

    return df


# ============================================================
# 14. SUPPORT / PAYMENT RISK
# ============================================================

def calculate_support_risk(df):

    support_score = min_max_scale(
        df["total_support_tickets"]
    )

    payment_score = (
        df["payment_failure_rate"]
        * 100
    )

    df["support_risk_score"] = (
        support_score * 0.60
        +
        payment_score * 0.40
    ).clip(0, 100)

    return df


# ============================================================
# 15. FINAL RISK SCORE
# ============================================================

def calculate_final_risk(df):

    """
    Final risk score:

        Churn       = 40%
        CLV Impact  = 15%
        Sentiment   = 15%
        Anomaly     = 10%
        Engagement  = 10%
        Support     = 10%

    Total = 100%
    """

    df["risk_score"] = (

        df["churn_risk_score"] * 0.40

        +

        df["clv_impact_score"] * 0.15

        +

        df["sentiment_risk_score"] * 0.15

        +

        df["anomaly_risk_score"] * 0.10

        +

        df["engagement_risk_score"] * 0.10

        +

        df["support_risk_score"] * 0.10

    ).clip(0, 100)

    return df


# ============================================================
# 16. RISK LEVEL
# ============================================================

def assign_risk_level(df):

    conditions = [

        df["risk_score"] >= 80,

        df["risk_score"] >= 60,

        df["risk_score"] >= 40

    ]

    choices = [
        "CRITICAL",
        "HIGH",
        "MEDIUM"
    ]

    df["risk_level"] = np.select(
        conditions,
        choices,
        default="LOW"
    )

    return df


# ============================================================
# 17. RISK REASON
# ============================================================

def generate_risk_reason(df):

    reasons = []

    for _, row in df.iterrows():

        customer_reasons = []

        if row["churn_risk_score"] >= 70:
            customer_reasons.append(
                "High churn probability"
            )

        if row["clv_impact_score"] >= 70:
            customer_reasons.append(
                "High customer value"
            )

        if row["sentiment_risk_score"] >= 70:
            customer_reasons.append(
                "Negative sentiment"
            )

        if row["anomaly_risk_score"] >= 70:
            customer_reasons.append(
                "Anomalous customer behavior"
            )

        if row["engagement_risk_score"] >= 70:
            customer_reasons.append(
                "Low engagement"
            )

        if row["support_risk_score"] >= 70:
            customer_reasons.append(
                "High support/payment risk"
            )

        if len(customer_reasons) == 0:
            customer_reasons.append(
                "No major risk signals"
            )

        reasons.append(
            "; ".join(customer_reasons)
        )

    df["risk_reason"] = reasons

    return df


# ============================================================
# 18. CREATE FINAL TABLE
# ============================================================

def create_final_table(df):

    columns = [

        "customer_id",

        "risk_score",

        "risk_level",

        "risk_reason",

        "churn_probability",

        "churn_risk_score",

        "predicted_churn",

        "model_name",

        "clv",

        "clv_impact_score",

        "sentiment_score",

        "sentiment_risk_score",

        "anomaly_score",

        "anomaly_flag",

        "anomaly_risk_score",

        "engagement_risk_score",

        "support_risk_score"

    ]

    result = df[columns].copy()

    result = result.sort_values(
        "risk_score",
        ascending=False
    )

    return result


# ============================================================
# 19. SAVE RESULTS
# ============================================================

def save_results(result):

    print("\nSaving customer risk results...")

    result.to_sql(
        "customer_risk",
        engine,
        if_exists="replace",
        index=False
    )

    print(
        "customer_risk table saved successfully."
    )


# ============================================================
# 20. SUMMARY
# ============================================================

def show_summary(result):

    print("\n")
    print("=" * 60)
    print("CUSTOMER RISK ENGINE SUMMARY")
    print("=" * 60)

    print(
        f"\nCustomers processed: "
        f"{len(result):,}"
    )

    print("\nRisk Level Distribution:")

    distribution = (
        result["risk_level"]
        .value_counts()
        .reindex(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            fill_value=0
        )
    )

    print(distribution)

    print(
        f"\nAverage Risk Score: "
        f"{result['risk_score'].mean():.2f}"
    )

    print(
        f"Maximum Risk Score: "
        f"{result['risk_score'].max():.2f}"
    )

    print("\nTop 10 Highest-Risk Customers:")

    print(
        result[
            [
                "customer_id",
                "risk_score",
                "risk_level",
                "risk_reason",
                "churn_probability",
                "clv"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print(
        "PHASE 13 CUSTOMER RISK ENGINE "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# 21. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOMER RISK ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    customer_360 = load_customer_360()

    churn = load_churn_predictions()

    clv = load_clv()

    sentiment = load_sentiment()

    anomalies = load_anomalies()

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    df = combine_data(
        customer_360,
        churn,
        clv,
        sentiment,
        anomalies
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = clean_data(df)

    # --------------------------------------------------------
    # RISK COMPONENTS
    # --------------------------------------------------------

    df = calculate_churn_risk(df)

    df = calculate_clv_impact(df)

    df = calculate_sentiment_risk(df)

    df = calculate_anomaly_risk(df)

    df = calculate_engagement_risk(df)

    df = calculate_support_risk(df)

    # --------------------------------------------------------
    # FINAL RISK
    # --------------------------------------------------------

    df = calculate_final_risk(df)

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    df = assign_risk_level(df)

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    df = generate_risk_reason(df)

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    result = create_final_table(df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(result)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    show_summary(result)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
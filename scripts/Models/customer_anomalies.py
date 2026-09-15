"""
Phase 12 - Customer Anomaly Detection

Detects unusual customer behavior using:
1. Isolation Forest
2. Z-score analysis
3. Statistical business thresholds

Input:
    customer_360 table in PostgreSQL

Output:
    customer_anomalies table in PostgreSQL
"""

import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# DATABASE CONNECTION
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# ============================================================
# LOAD CUSTOMER 360 DATA
# ============================================================

def load_customer_data():

    print("\nLoading customer 360 data...")

    query = """
        SELECT
            customer_id,
            total_spend,
            total_transactions,
            payment_failure_rate,
            total_logins,
            total_session_minutes,
            total_support_tickets,
            average_session_minutes,
            support_burden,
            average_features_used
        FROM customer_360
    """

    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df):,} customers.")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("\nPreparing anomaly detection features...")

    features = [
        "total_spend",
        "total_transactions",
        "payment_failure_rate",
        "total_logins",
        "total_session_minutes",
        "total_support_tickets",
        "average_session_minutes",
        "support_burden",
        "average_features_used"
    ]

    X = df[features].copy()

    # Convert everything to numeric
    for column in features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Replace missing/infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(X.median())

    return X, features


# ============================================================
# ISOLATION FOREST
# ============================================================

def isolation_forest_detection(X):

    print("\nRunning Isolation Forest...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    predictions = model.fit_predict(X_scaled)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly

    anomaly_flag = np.where(predictions == -1, 1, 0)

    # Convert decision function into an easier-to-read score.
    # Lower decision_function values mean more anomalous.
    raw_score = model.decision_function(X_scaled)

    # Higher anomaly score = more anomalous
    anomaly_score = -raw_score

    return anomaly_flag, anomaly_score


# ============================================================
# Z-SCORE DETECTION
# ============================================================

def z_score_detection(X):

    print("\nRunning Z-score analysis...")

    z_scores = pd.DataFrame(index=X.index)

    for column in X.columns:

        mean = X[column].mean()
        std = X[column].std()

        if std == 0 or pd.isna(std):
            z_scores[column] = 0
        else:
            z_scores[column] = (X[column] - mean) / std

    # Absolute z-score >= 3 means extreme observation
    z_anomaly_count = (z_scores.abs() >= 3).sum(axis=1)

    z_anomaly_flag = np.where(
        z_anomaly_count > 0,
        1,
        0
    )

    return z_anomaly_flag, z_anomaly_count


# ============================================================
# STATISTICAL / BUSINESS THRESHOLDS
# ============================================================

def statistical_threshold_detection(df):

    print("\nApplying statistical/business thresholds...")

    flags = pd.DataFrame(index=df.index)

    # Very high payment failure rate
    flags["payment_failure_anomaly"] = (
        df["payment_failure_rate"] >= 0.50
    ).astype(int)

    # Very high support ticket volume
    support_threshold = df["total_support_tickets"].quantile(0.95)

    flags["support_anomaly"] = (
        df["total_support_tickets"] >= support_threshold
    ).astype(int)

    # Very low spending
    spend_threshold = df["total_spend"].quantile(0.05)

    flags["low_spend_anomaly"] = (
        df["total_spend"] <= spend_threshold
    ).astype(int)

    # Very high session activity
    session_threshold = df["total_session_minutes"].quantile(0.95)

    flags["usage_anomaly"] = (
        df["total_session_minutes"] >= session_threshold
    ).astype(int)

    # Very high transaction volume
    transaction_threshold = df["total_transactions"].quantile(0.95)

    flags["transaction_anomaly"] = (
        df["total_transactions"] >= transaction_threshold
    ).astype(int)

    statistical_anomaly_count = flags.sum(axis=1)

    statistical_anomaly_flag = np.where(
        statistical_anomaly_count > 0,
        1,
        0
    )

    return (
        statistical_anomaly_flag,
        statistical_anomaly_count,
        flags
    )


# ============================================================
# BUILD FINAL ANOMALY TABLE
# ============================================================

def build_anomaly_table(
    df,
    isolation_flag,
    isolation_score,
    z_flag,
    z_count,
    statistical_flag,
    statistical_count,
    statistical_flags
):

    print("\nCombining anomaly detection results...")

    result = df[["customer_id"]].copy()

    result["isolation_forest_flag"] = isolation_flag
    result["isolation_forest_score"] = isolation_score

    result["z_score_flag"] = z_flag
    result["z_score_anomaly_count"] = z_count

    result["statistical_anomaly_flag"] = statistical_flag
    result["statistical_anomaly_count"] = statistical_count

    # Add individual business-rule flags
    result["payment_failure_anomaly"] = (
        statistical_flags["payment_failure_anomaly"].values
    )

    result["support_anomaly"] = (
        statistical_flags["support_anomaly"].values
    )

    result["low_spend_anomaly"] = (
        statistical_flags["low_spend_anomaly"].values
    )

    result["usage_anomaly"] = (
        statistical_flags["usage_anomaly"].values
    )

    result["transaction_anomaly"] = (
        statistical_flags["transaction_anomaly"].values
    )

    # --------------------------------------------------------
    # FINAL ANOMALY FLAG
    # --------------------------------------------------------
    #
    # Customer is considered anomalous if:
    # Isolation Forest OR Z-score OR statistical rules
    # detect unusual behavior.
    #
    result["anomaly_flag"] = (
        (
            (result["isolation_forest_flag"] == 1)
            |
            (result["z_score_flag"] == 1)
            |
            (result["statistical_anomaly_flag"] == 1)
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # COMBINED ANOMALY SCORE
    # --------------------------------------------------------

    result["anomaly_score"] = (
        result["isolation_forest_score"]
        + (result["z_score_anomaly_count"] * 0.20)
        + (result["statistical_anomaly_count"] * 0.20)
    )

    # --------------------------------------------------------
    # ANOMALY CATEGORY
    # --------------------------------------------------------

    result["anomaly_category"] = np.select(
        [
            result["anomaly_score"] >= 0.70,
            result["anomaly_score"] >= 0.40,
            result["anomaly_flag"] == 1
        ],
        [
            "High",
            "Medium",
            "Low"
        ],
        default="Normal"
    )

    return result


# ============================================================
# SAVE TO POSTGRESQL
# ============================================================

def save_results(result):

    print("\nSaving anomaly results to PostgreSQL...")

    result.to_sql(
        "customer_anomalies",
        engine,
        if_exists="replace",
        index=False
    )

    print("customer_anomalies table saved successfully.")


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def show_summary(result):

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 60)

    print(f"Total customers analyzed: {len(result):,}")

    print("\nIsolation Forest anomalies:")
    print(
        result["isolation_forest_flag"]
        .value_counts()
        .rename(index={
            0: "Normal",
            1: "Anomaly"
        })
    )

    print("\nZ-score anomalies:")
    print(
        result["z_score_flag"]
        .value_counts()
        .rename(index={
            0: "Normal",
            1: "Anomaly"
        })
    )

    print("\nStatistical anomalies:")
    print(
        result["statistical_anomaly_flag"]
        .value_counts()
        .rename(index={
            0: "Normal",
            1: "Anomaly"
        })
    )

    print("\nFinal anomaly status:")
    print(
        result["anomaly_flag"]
        .value_counts()
        .rename(index={
            0: "Normal",
            1: "Anomaly"
        })
    )

    print("\nAnomaly categories:")
    print(result["anomaly_category"].value_counts())

    print("\nTop 10 most anomalous customers:")

    top_customers = result[
        [
            "customer_id",
            "anomaly_score",
            "anomaly_category",
            "isolation_forest_flag",
            "z_score_flag",
            "statistical_anomaly_flag"
        ]
    ].sort_values(
        "anomaly_score",
        ascending=False
    ).head(10)

    print(top_customers.to_string(index=False))

    print("\n" + "=" * 60)
    print("PHASE 12 ANOMALY DETECTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOMER ANOMALY DETECTION PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_customer_data()

    # 2. Prepare features
    X, features = prepare_features(df)

    # 3. Isolation Forest
    isolation_flag, isolation_score = (
        isolation_forest_detection(X)
    )

    # 4. Z-score
    z_flag, z_count = z_score_detection(X)

    # 5. Statistical thresholds
    (
        statistical_flag,
        statistical_count,
        statistical_flags
    ) = statistical_threshold_detection(df)

    # 6. Build final table
    result = build_anomaly_table(
        df,
        isolation_flag,
        isolation_score,
        z_flag,
        z_count,
        statistical_flag,
        statistical_count,
        statistical_flags
    )

    # 7. Save
    save_results(result)

    # 8. Display summary
    show_summary(result)


if __name__ == "__main__":
    main()
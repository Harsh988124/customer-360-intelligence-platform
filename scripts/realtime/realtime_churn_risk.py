import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "customer_intelligence")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURES = [
    "age",
    "monthly_price",
    "subscription_count",
    "auto_renew_count",
    "total_transactions",
    "successful_transactions",
    "failed_transactions",
    "total_spend",
    "average_spend_per_transaction",
    "transaction_success_rate",
    "payment_failure_rate",
    "usage_records",
    "active_days",
    "total_logins",
    "total_session_minutes",
    "total_features_used",
    "average_features_used",
    "average_session_minutes",
    "total_support_tickets",
    "average_resolution_hours",
    "negative_tickets",
    "negative_ticket_rate",
    "tenure_days",
    "tenure_months",
    "support_burden"
]


# =========================================================
# LOAD HISTORICAL TRAINING DATA
# =========================================================

print("Loading historical Customer 360 data...")

training_df = pd.read_sql(
    """
    SELECT *
    FROM customer_360
    """,
    engine
)

print(
    f"Historical customers loaded: "
    f"{len(training_df):,}"
)


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_training_columns = FEATURES + ["churned"]

missing_columns = [
    column
    for column in required_training_columns
    if column not in training_df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns in customer_360: "
        + ", ".join(missing_columns)
    )


# =========================================================
# PREPARE TRAINING DATA
# =========================================================

X_train = training_df[FEATURES].copy()

y_train = training_df["churned"].astype(int)


numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = [
    column
    for column in FEATURES
    if column not in numeric_features
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# =========================================================
# TRAIN MODEL
# =========================================================

print("Training Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# =========================================================
# LOAD REAL-TIME CUSTOMER 360
# =========================================================

print("\nLoading real-time Customer 360...")

realtime_df = pd.read_sql(
    """
    SELECT *
    FROM realtime_customer_360
    """,
    engine
)

print(
    f"Real-time customers loaded: "
    f"{len(realtime_df):,}"
)


# =========================================================
# MERGE REAL-TIME EVENTS WITH CUSTOMER 360
# =========================================================

print("Combining historical and real-time metrics...")


# Start with historical Customer 360
live_df = training_df.copy()


# ---------------------------------------------------------
# Real-time login updates
# ---------------------------------------------------------

if "realtime_logins" in realtime_df.columns:

    live_df = live_df.merge(
        realtime_df[
            [
                "customer_id",
                "realtime_logins"
            ]
        ],
        on="customer_id",
        how="left"
    )

    live_df["realtime_logins"] = (
        live_df["realtime_logins"]
        .fillna(0)
    )

    live_df["total_logins"] = (
        live_df["total_logins"].fillna(0)
        + live_df["realtime_logins"]
    )

    live_df.drop(
        columns=["realtime_logins"],
        inplace=True
    )


# ---------------------------------------------------------
# Real-time transactions
# ---------------------------------------------------------

if "realtime_transactions" in realtime_df.columns:

    live_df = live_df.merge(
        realtime_df[
            [
                "customer_id",
                "realtime_transactions"
            ]
        ],
        on="customer_id",
        how="left"
    )

    live_df["realtime_transactions"] = (
        live_df["realtime_transactions"]
        .fillna(0)
    )

    live_df["total_transactions"] = (
        live_df["total_transactions"].fillna(0)
        + live_df["realtime_transactions"]
    )

    live_df["successful_transactions"] = (
        live_df["successful_transactions"].fillna(0)
        + live_df["realtime_transactions"]
    )

    live_df.drop(
        columns=["realtime_transactions"],
        inplace=True
    )


# ---------------------------------------------------------
# Real-time spend
# ---------------------------------------------------------

if "realtime_spend" in realtime_df.columns:

    live_df = live_df.merge(
        realtime_df[
            [
                "customer_id",
                "realtime_spend"
            ]
        ],
        on="customer_id",
        how="left"
    )

    live_df["realtime_spend"] = (
        live_df["realtime_spend"]
        .fillna(0)
    )

    live_df["total_spend"] = (
        live_df["total_spend"].fillna(0)
        + live_df["realtime_spend"]
    )

    live_df.drop(
        columns=["realtime_spend"],
        inplace=True
    )


# ---------------------------------------------------------
# Real-time payment failures
# ---------------------------------------------------------

if "realtime_payment_failures" in realtime_df.columns:

    live_df = live_df.merge(
        realtime_df[
            [
                "customer_id",
                "realtime_payment_failures"
            ]
        ],
        on="customer_id",
        how="left"
    )

    live_df["realtime_payment_failures"] = (
        live_df["realtime_payment_failures"]
        .fillna(0)
    )

    live_df["failed_transactions"] = (
        live_df["failed_transactions"].fillna(0)
        + live_df["realtime_payment_failures"]
    )

    live_df["total_transactions"] = (
        live_df["total_transactions"].fillna(0)
        + live_df["realtime_payment_failures"]
    )

    live_df.drop(
        columns=["realtime_payment_failures"],
        inplace=True
    )


# ---------------------------------------------------------
# Real-time support tickets
# ---------------------------------------------------------

if "realtime_support_tickets" in realtime_df.columns:

    live_df = live_df.merge(
        realtime_df[
            [
                "customer_id",
                "realtime_support_tickets"
            ]
        ],
        on="customer_id",
        how="left"
    )

    live_df["realtime_support_tickets"] = (
        live_df["realtime_support_tickets"]
        .fillna(0)
    )

    live_df["total_support_tickets"] = (
        live_df["total_support_tickets"].fillna(0)
        + live_df["realtime_support_tickets"]
    )

    live_df.drop(
        columns=["realtime_support_tickets"],
        inplace=True
    )


# =========================================================
# RECALCULATE DERIVED FEATURES
# =========================================================

live_df["average_spend_per_transaction"] = np.where(
    live_df["total_transactions"] > 0,
    live_df["total_spend"]
    / live_df["total_transactions"],
    0
)


live_df["transaction_success_rate"] = np.where(
    live_df["total_transactions"] > 0,
    live_df["successful_transactions"]
    / live_df["total_transactions"],
    0
)


live_df["payment_failure_rate"] = np.where(
    live_df["total_transactions"] > 0,
    live_df["failed_transactions"]
    / live_df["total_transactions"],
    0
)


live_df["negative_ticket_rate"] = np.where(
    live_df["total_support_tickets"] > 0,
    live_df["negative_tickets"]
    / live_df["total_support_tickets"],
    0
)


live_df["support_burden"] = np.where(
    live_df["tenure_months"] > 0,
    live_df["total_support_tickets"]
    / live_df["tenure_months"],
    live_df["total_support_tickets"]
)


# =========================================================
# PREDICT LIVE CHURN
# =========================================================

print("\nCalculating real-time churn probability...")

X_live = live_df[FEATURES].copy()

live_df["churn_probability_ml"] = model.predict_proba(
    X_live
)[:, 1]

live_df["predicted_churn"] = (
    live_df["churn_probability_ml"] >= 0.50
).astype(int)

live_df["model_name"] = (
    "Logistic Regression - Real-Time"
)


# =========================================================
# CREATE REAL-TIME CHURN TABLE
# =========================================================

churn_output = live_df[
    [
        "customer_id",
        "churn_probability_ml",
        "predicted_churn",
        "model_name"
    ]
].copy()


print("Saving real-time churn predictions...")


churn_output.to_sql(
    "realtime_churn_predictions",
    engine,
    if_exists="replace",
    index=False
)


# =========================================================
# LOAD CLV
# =========================================================

print("Loading customer CLV...")

clv_df = pd.read_sql(
    """
    SELECT *
    FROM customer_clv
    """,
    engine
)


# Find CLV column
clv_candidates = [
    "estimated_clv",
    "customer_lifetime_value",
    "clv"
]

clv_column = None

for column in clv_candidates:

    if column in clv_df.columns:

        clv_column = column
        break


if clv_column is None:

    raise ValueError(
        "Could not find CLV column."
    )


clv_small = clv_df[
    [
        "customer_id",
        clv_column
    ]
].copy()

clv_small.rename(
    columns={
        clv_column: "estimated_clv"
    },
    inplace=True
)


# =========================================================
# REAL-TIME RISK ENGINE
# =========================================================

print("Calculating real-time customer risk...")


risk_df = churn_output.merge(
    clv_small,
    on="customer_id",
    how="left"
)


# ---------------------------------------------------------
# Churn risk
# ---------------------------------------------------------

risk_df["churn_risk_score"] = (
    risk_df["churn_probability_ml"] * 100
)


# ---------------------------------------------------------
# CLV impact
# ---------------------------------------------------------

clv_min = risk_df["estimated_clv"].min()
clv_max = risk_df["estimated_clv"].max()

if clv_max > clv_min:

    risk_df["clv_impact_score"] = (
        (risk_df["estimated_clv"] - clv_min)
        / (clv_max - clv_min)
        * 100
    )

else:

    risk_df["clv_impact_score"] = 0


# ---------------------------------------------------------
# Get sentiment
# ---------------------------------------------------------

sentiment_df = pd.read_sql(
    """
    SELECT
        customer_id,
        sentiment_score,
        negative_sentiment_rate
    FROM customer_sentiment
    """,
    engine
)


risk_df = risk_df.merge(
    sentiment_df,
    on="customer_id",
    how="left"
)


risk_df["sentiment_score"] = (
    risk_df["sentiment_score"]
    .fillna(0)
)

risk_df["negative_sentiment_rate"] = (
    risk_df["negative_sentiment_rate"]
    .fillna(0)
)


risk_df["sentiment_risk_score"] = (
    (1 - risk_df["sentiment_score"])
    / 2
    * 100
)


# ---------------------------------------------------------
# Get anomaly information
# ---------------------------------------------------------

anomaly_df = pd.read_sql(
    """
    SELECT
        customer_id,
        anomaly_flag,
        anomaly_category,
        anomaly_score
    FROM customer_anomalies
    """,
    engine
)


risk_df = risk_df.merge(
    anomaly_df,
    on="customer_id",
    how="left"
)


risk_df["anomaly_score"] = (
    risk_df["anomaly_score"]
    .fillna(0)
)

risk_df["anomaly_flag"] = (
    risk_df["anomaly_flag"]
    .fillna(False)
)


anomaly_min = risk_df["anomaly_score"].min()
anomaly_max = risk_df["anomaly_score"].max()

if anomaly_max > anomaly_min:

    risk_df["anomaly_risk_score"] = (
        (risk_df["anomaly_score"] - anomaly_min)
        / (anomaly_max - anomaly_min)
        * 100
    )

else:

    risk_df["anomaly_risk_score"] = 0


# =========================================================
# ENGAGEMENT RISK
# =========================================================

engagement = live_df[
    [
        "customer_id",
        "total_logins",
        "total_session_minutes"
    ]
].copy()


risk_df = risk_df.merge(
    engagement,
    on="customer_id",
    how="left"
)


login_min = risk_df["total_logins"].min()
login_max = risk_df["total_logins"].max()

session_min = risk_df["total_session_minutes"].min()
session_max = risk_df["total_session_minutes"].max()


if login_max > login_min:

    login_risk = (
        1
        -
        (
            (risk_df["total_logins"] - login_min)
            /
            (login_max - login_min)
        )
    ) * 100

else:

    login_risk = 0


if session_max > session_min:

    session_risk = (
        1
        -
        (
            (
                risk_df["total_session_minutes"]
                - session_min
            )
            /
            (session_max - session_min)
        )
    ) * 100

else:

    session_risk = 0


risk_df["engagement_risk_score"] = (
    login_risk * 0.5
    +
    session_risk * 0.5
)


# =========================================================
# SUPPORT RISK
# =========================================================

support_df = live_df[
    [
        "customer_id",
        "total_support_tickets",
        "payment_failure_rate"
    ]
].copy()


risk_df = risk_df.merge(
    support_df,
    on="customer_id",
    how="left"
)


support_min = risk_df["total_support_tickets"].min()
support_max = risk_df["total_support_tickets"].max()


if support_max > support_min:

    support_ticket_risk = (
        (
            risk_df["total_support_tickets"]
            - support_min
        )
        /
        (
            support_max
            - support_min
        )
        * 100
    )

else:

    support_ticket_risk = 0


risk_df["support_risk_score"] = (
    support_ticket_risk * 0.60
    +
    risk_df["payment_failure_rate"] * 100 * 0.40
)


# =========================================================
# FINAL RISK SCORE
# =========================================================

risk_df["risk_score"] = (
    risk_df["churn_risk_score"] * 0.40
    +
    risk_df["clv_impact_score"] * 0.15
    +
    risk_df["sentiment_risk_score"] * 0.15
    +
    risk_df["anomaly_risk_score"] * 0.10
    +
    risk_df["engagement_risk_score"] * 0.10
    +
    risk_df["support_risk_score"] * 0.10
)


# =========================================================
# RISK LEVEL
# =========================================================

def risk_level(score):

    if score >= 80:
        return "CRITICAL"

    elif score >= 60:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


risk_df["risk_level"] = (
    risk_df["risk_score"]
    .apply(risk_level)
)


# =========================================================
# RISK REASON
# =========================================================

def generate_reason(row):

    reasons = []

    if row["churn_probability_ml"] >= 0.80:
        reasons.append("Very high churn probability")

    elif row["churn_probability_ml"] >= 0.60:
        reasons.append("High churn probability")

    if row["sentiment_score"] < -0.30:
        reasons.append("Negative customer sentiment")

    if row["payment_failure_rate"] >= 0.30:
        reasons.append("High payment failure rate")

    if row["total_support_tickets"] >= 10:
        reasons.append("High support activity")

    if row["engagement_risk_score"] >= 60:
        reasons.append("Low customer engagement")

    if row["anomaly_flag"]:
        reasons.append("Unusual customer behavior")

    if not reasons:
        reasons.append("Normal customer behavior")

    return "; ".join(reasons)


risk_df["risk_reason"] = risk_df.apply(
    generate_reason,
    axis=1
)


# =========================================================
# SAVE REAL-TIME RISK TABLE
# =========================================================

risk_output = risk_df[
    [
        "customer_id",
        "risk_score",
        "risk_level",
        "estimated_clv",
        "churn_probability_ml",
        "predicted_churn",
        "sentiment_score",
        "negative_sentiment_rate",
        "anomaly_flag",
        "anomaly_category",
        "anomaly_score",
        "engagement_risk_score",
        "support_risk_score",
        "risk_reason"
    ]
].copy()


print("Saving real-time customer risk...")

risk_output.to_sql(
    "realtime_customer_risk",
    engine,
    if_exists="replace",
    index=False
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("PHASE 19.3 - REAL-TIME CHURN + RISK COMPLETED")
print("=" * 60)

print(
    f"Customers processed: "
    f"{len(risk_output):,}"
)

print(
    f"Average churn probability: "
    f"{risk_output['churn_probability_ml'].mean():.4f}"
)

print(
    f"Average risk score: "
    f"{risk_output['risk_score'].mean():.2f}"
)

print("\nRisk distribution:")

print(
    risk_output["risk_level"]
    .value_counts()
)


print("\nTop 10 highest-risk customers:")

top_risk = risk_output.sort_values(
    "risk_score",
    ascending=False
).head(10)


print(
    top_risk[
        [
            "customer_id",
            "risk_score",
            "risk_level",
            "churn_probability_ml",
            "estimated_clv",
            "risk_reason"
        ]
    ].to_string(index=False)
)


print("\nTables created successfully:")

print("1. realtime_churn_predictions")
print("2. realtime_customer_risk")

print("\nPhase 19.3 finished.")
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("Database environment variables are missing.")


# ============================================================
# 2. DATABASE CONNECTION
# ============================================================

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# ============================================================
# 3. LOAD CUSTOMER RISK DATA
# ============================================================

print("Loading customer risk data...")

# Temporary debug code to see column names
columns_df = pd.read_sql("SELECT * FROM customer_sentiment LIMIT 0", engine)
print("Available columns in DB:", columns_df.columns.tolist())

risk_query = """
SELECT *
FROM customer_risk
"""

risk_df = pd.read_sql(risk_query, engine)

print(f"Customers loaded: {len(risk_df)}")


# ============================================================
# 4. LOAD ADDITIONAL CUSTOMER SIGNALS
# ============================================================

clv_query = """
SELECT
    customer_id,
    estimated_clv
FROM customer_clv
"""

sentiment_query = """
SELECT
    customer_id,
    sentiment_score,
    negative_ticket_count AS negative_count, 
    total_sentiment_tickets AS total_tickets
FROM customer_sentiment
"""

churn_query = """
SELECT
    customer_id,
    churn_probability_ml,
    predicted_churn
FROM churn_predictions
"""

anomaly_query = """
SELECT
    customer_id,
    anomaly_flag,
    anomaly_category,
    anomaly_score
FROM customer_anomalies
"""


clv_df = pd.read_sql(clv_query, engine)
sentiment_df = pd.read_sql(sentiment_query, engine)
churn_df = pd.read_sql(churn_query, engine)
anomaly_df = pd.read_sql(anomaly_query, engine)

# Calculate negative sentiment rate
sentiment_df["negative_rate"] = (
    sentiment_df["negative_count"]
    / sentiment_df["total_tickets"].replace(0, 1)
)


# ============================================================
# 5. MERGE DATA
# ============================================================

# ============================================================
# 5. MERGE DATA
# ============================================================

df = risk_df.copy()

# A helper function to prevent _x and _y column collisions during merge
def safe_merge(left_df, right_df, merge_on="customer_id"):
    # Find overlapping columns (excluding the merge key)
    overlap = set(left_df.columns).intersection(set(right_df.columns)) - {merge_on}
    # Drop the overlapping columns from the right dataframe so we don't get duplicates
    right_df_clean = right_df.drop(columns=list(overlap))
    return left_df.merge(right_df_clean, on=merge_on, how="left")

df = safe_merge(df, clv_df)
df = safe_merge(df, sentiment_df)
df = safe_merge(df, churn_df)
df = safe_merge(df, anomaly_df)


# ============================================================
# 6. CLEAN DATA
# ============================================================

numeric_columns = [
    "risk_score",
    "estimated_clv",
    "sentiment_score",
    "negative_rate",
    "churn_probability_ml",
    "anomaly_score"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

df["risk_score"] = df["risk_score"].fillna(0)

df["estimated_clv"] = df["estimated_clv"].fillna(0)

df["sentiment_score"] = df["sentiment_score"].fillna(0)

df["negative_rate"] = df["negative_rate"].fillna(0)

df["churn_probability_ml"] = df["churn_probability_ml"].fillna(0)

df["anomaly_score"] = df["anomaly_score"].fillna(0)

df["anomaly_flag"] = df["anomaly_flag"].fillna(False)


# ============================================================
# 7. CALCULATE CLV SEGMENT
# ============================================================

clv_75 = df["estimated_clv"].quantile(0.75)
clv_50 = df["estimated_clv"].quantile(0.50)

def get_clv_segment(clv):

    if clv >= clv_75:
        return "High CLV"

    elif clv >= clv_50:
        return "Medium CLV"

    else:
        return "Low CLV"


df["clv_segment"] = df["estimated_clv"].apply(
    get_clv_segment
)


# ============================================================
# 8. NEXT BEST ACTION LOGIC
# ============================================================

def recommend_action(row):

    risk_level = str(row.get("risk_level", "LOW")).upper()

    risk_score = row["risk_score"]

    churn_probability = row["churn_probability_ml"]

    sentiment_score = row["sentiment_score"]

    negative_rate = row["negative_rate"]

    clv_segment = row["clv_segment"]

    anomaly_flag = bool(row["anomaly_flag"])


    # --------------------------------------------------------
    # RULE 1: CRITICAL + HIGH CLV
    # --------------------------------------------------------

    if (
        risk_level == "CRITICAL"
        and clv_segment == "High CLV"
        and churn_probability >= 0.70
    ):

        return pd.Series({
            "recommended_action":
                "Executive Retention Outreach",

            "action_priority":
                "P1 - Critical",

            "action_reason":
                "Critical-risk high-value customer with high churn probability."
        })


    # --------------------------------------------------------
    # RULE 2: NEGATIVE CUSTOMER EXPERIENCE
    # --------------------------------------------------------

    if (
        churn_probability >= 0.70
        and (
            sentiment_score < -0.30
            or negative_rate >= 0.60
        )
    ):

        return pd.Series({
            "recommended_action":
                "Priority Support Intervention",

            "action_priority":
                "P1 - Critical",

            "action_reason":
                "High churn probability combined with negative customer sentiment."
        })


    # --------------------------------------------------------
    # RULE 3: ANOMALOUS BEHAVIOR
    # --------------------------------------------------------

    if (
        anomaly_flag
        and risk_level in ["CRITICAL", "HIGH"]
    ):

        return pd.Series({
            "recommended_action":
                "Account Behavior Review",

            "action_priority":
                "P1 - Critical",

            "action_reason":
                "Customer shows anomalous behavioral or transaction patterns."
        })


    # --------------------------------------------------------
    # RULE 4: HIGH CHURN + HIGH CLV
    # --------------------------------------------------------

    if (
        churn_probability >= 0.70
        and clv_segment == "High CLV"
    ):

        return pd.Series({
            "recommended_action":
                "Personalized Retention Offer",

            "action_priority":
                "P1 - High",

            "action_reason":
                "High-value customer has a high probability of churn."
        })


    # --------------------------------------------------------
    # RULE 5: HIGH CHURN
    # --------------------------------------------------------

    if churn_probability >= 0.70:

        return pd.Series({
            "recommended_action":
                "Retention Campaign",

            "action_priority":
                "P2 - High",

            "action_reason":
                "Customer has a high predicted probability of churn."
        })


    # --------------------------------------------------------
    # RULE 6: MEDIUM RISK + NEGATIVE SENTIMENT
    # --------------------------------------------------------

    if (
        risk_level == "MEDIUM"
        and (
            sentiment_score < -0.20
            or negative_rate >= 0.50
        )
    ):

        return pd.Series({
            "recommended_action":
                "Customer Experience Follow-up",

            "action_priority":
                "P2 - Medium",

            "action_reason":
                "Medium-risk customer shows negative support sentiment."
        })


    # --------------------------------------------------------
    # RULE 7: PAYMENT / ACCOUNT RISK
    # --------------------------------------------------------

    if (
        risk_level in ["HIGH", "CRITICAL"]
        and anomaly_flag
    ):

        return pd.Series({
            "recommended_action":
                "Payment and Account Review",

            "action_priority":
                "P2 - High",

            "action_reason":
                "Customer has elevated risk with an anomaly signal."
        })


    # --------------------------------------------------------
    # RULE 8: HIGH CLV BUT LOW RISK
    # --------------------------------------------------------

    if (
        clv_segment == "High CLV"
        and risk_level == "LOW"
    ):

        return pd.Series({
            "recommended_action":
                "VIP Loyalty Engagement",

            "action_priority":
                "P3 - Normal",

            "action_reason":
                "High-value customer is currently healthy and suitable for loyalty engagement."
        })


    # --------------------------------------------------------
    # RULE 9: MEDIUM RISK
    # --------------------------------------------------------

    if risk_level == "MEDIUM":

        return pd.Series({
            "recommended_action":
                "Personalized Retention Email",

            "action_priority":
                "P3 - Normal",

            "action_reason":
                "Customer has moderate overall retention risk."
        })


    # --------------------------------------------------------
    # RULE 10: LOW RISK
    # --------------------------------------------------------

    return pd.Series({
        "recommended_action":
            "Monitor Customer",

        "action_priority":
            "P4 - Low",

        "action_reason":
            "Customer currently shows low retention risk."
    })


# ============================================================
# 9. APPLY RECOMMENDATIONS
# ============================================================

print("Generating next best actions...")

recommendations = df.apply(
    recommend_action,
    axis=1
)

df = pd.concat(
    [df, recommendations],
    axis=1
)


# ============================================================
# 10. CREATE FINAL OUTPUT
# ============================================================

output_columns = [
    "customer_id",
    "risk_score",
    "risk_level",
    "estimated_clv",
    "clv_segment",
    "churn_probability_ml",
    "predicted_churn",
    "sentiment_score",
    "negative_rate",
    "anomaly_flag",
    "anomaly_category",
    "anomaly_score",
    "recommended_action",
    "action_priority",
    "action_reason"
]

output_columns = [
    col for col in output_columns
    if col in df.columns
]

nba_df = df[output_columns].copy()


# ============================================================
# 11. SAVE TO POSTGRESQL
# ============================================================

print("Saving Next Best Action table...")

with engine.begin() as connection:

    connection.execute(
        text("DROP TABLE IF EXISTS next_best_action")
    )

nba_df.to_sql(
    "next_best_action",
    engine,
    if_exists="replace",
    index=False
)


# ============================================================
# 12. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PHASE 14 - NEXT BEST ACTION COMPLETED")
print("=" * 60)

print(f"\nTotal customers: {len(nba_df):,}")

print("\nRecommended actions:")

print(
    nba_df["recommended_action"]
    .value_counts()
    .to_string()
)

print("\nAction priorities:")

print(
    nba_df["action_priority"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSample recommendations:")

print(
    nba_df[
        [
            "customer_id",
            "risk_level",
            "estimated_clv",
            "churn_probability_ml",
            "recommended_action",
            "action_priority"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nTable created successfully:")
print("next_best_action")


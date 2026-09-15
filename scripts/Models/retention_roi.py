import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Loading Retention ROI data...")


# ---------------------------------------------------------
# 2. Load Revenue-at-Risk
# ---------------------------------------------------------

revenue_df = pd.read_sql(
    """
    SELECT
        customer_id,
        estimated_clv,
        churn_probability_ml,
        revenue_at_risk,
        revenue_risk_band,
        revenue_priority
    FROM revenue_at_risk
    """,
    engine
)

print(f"Revenue-at-Risk records loaded: {len(revenue_df):,}")


# ---------------------------------------------------------
# 3. Load Next Best Action
# ---------------------------------------------------------

nba_df = pd.read_sql(
    """
    SELECT
        customer_id,
        recommended_action,
        action_priority
    FROM next_best_action
    """,
    engine
)

print(f"Next Best Action records loaded: {len(nba_df):,}")


# ---------------------------------------------------------
# 4. Merge data
# ---------------------------------------------------------

df = revenue_df.merge(
    nba_df,
    on="customer_id",
    how="left"
)

print(f"Customers after merge: {len(df):,}")


# ---------------------------------------------------------
# 5. Clean numeric columns
# ---------------------------------------------------------

numeric_columns = [
    "estimated_clv",
    "churn_probability_ml",
    "revenue_at_risk"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)

df["churn_probability_ml"] = df[
    "churn_probability_ml"
].clip(0, 1)

df["estimated_clv"] = df[
    "estimated_clv"
].clip(lower=0)

df["revenue_at_risk"] = df[
    "revenue_at_risk"
].clip(lower=0)


# ---------------------------------------------------------
# 6. Define retention strategy
# ---------------------------------------------------------

def retention_strategy(row):

    probability = row["churn_probability_ml"]
    risk_band = row["revenue_risk_band"]
    action = row["recommended_action"]

    if probability >= 0.80 and risk_band == "Critical":
        return "High-Touch Retention"

    elif probability >= 0.60:
        return "Targeted Retention Campaign"

    elif probability >= 0.40:
        return "Personalized Engagement"

    else:
        return "Low-Cost Engagement"


df["retention_strategy"] = df.apply(
    retention_strategy,
    axis=1
)


# ---------------------------------------------------------
# 7. Assign retention cost
# ---------------------------------------------------------

# Illustrative business assumptions.
# These are not actual company costs.

def retention_cost(strategy):

    if strategy == "High-Touch Retention":
        return 3000.0

    elif strategy == "Targeted Retention Campaign":
        return 1500.0

    elif strategy == "Personalized Engagement":
        return 750.0

    else:
        return 250.0


df["retention_cost"] = df[
    "retention_strategy"
].apply(retention_cost)


# ---------------------------------------------------------
# 8. Assign expected recovery rate
# ---------------------------------------------------------

# Illustrative assumptions:
#
# High-Touch Retention       -> 70%
# Targeted Campaign         -> 60%
# Personalized Engagement   -> 50%
# Low-Cost Engagement       -> 30%

def recovery_rate(strategy):

    if strategy == "High-Touch Retention":
        return 0.70

    elif strategy == "Targeted Retention Campaign":
        return 0.60

    elif strategy == "Personalized Engagement":
        return 0.50

    else:
        return 0.30


df["expected_recovery_rate"] = df[
    "retention_strategy"
].apply(recovery_rate)


# ---------------------------------------------------------
# 9. Calculate expected recovered value
# ---------------------------------------------------------

df["expected_recovered_value"] = (
    df["revenue_at_risk"]
    * df["expected_recovery_rate"]
)


# ---------------------------------------------------------
# 10. Calculate net benefit
# ---------------------------------------------------------

df["expected_net_benefit"] = (
    df["expected_recovered_value"]
    - df["retention_cost"]
)


# ---------------------------------------------------------
# 11. Calculate ROI
# ---------------------------------------------------------

df["retention_roi"] = np.where(
    df["retention_cost"] > 0,
    df["expected_net_benefit"]
    / df["retention_cost"],
    0
)


# ROI percentage

df["retention_roi_percentage"] = (
    df["retention_roi"] * 100
)


# ---------------------------------------------------------
# 12. ROI classification
# ---------------------------------------------------------

def roi_category(roi):

    if roi >= 5:
        return "Excellent"

    elif roi >= 2:
        return "Strong"

    elif roi >= 1:
        return "Positive"

    elif roi >= 0:
        return "Break-even"

    else:
        return "Negative"


df["roi_category"] = df[
    "retention_roi"
].apply(roi_category)


# ---------------------------------------------------------
# 13. Retention decision
# ---------------------------------------------------------

def retention_decision(row):

    if row["expected_net_benefit"] <= 0:
        return "Do Not Retain"

    elif row["retention_roi"] >= 5:
        return "Immediate Retention"

    elif row["retention_roi"] >= 2:
        return "Prioritize Retention"

    elif row["retention_roi"] >= 1:
        return "Consider Retention"

    else:
        return "Monitor"


df["retention_decision"] = df.apply(
    retention_decision,
    axis=1
)


# ---------------------------------------------------------
# 14. ROI priority
# ---------------------------------------------------------

def roi_priority(row):

    if (
        row["retention_decision"] == "Immediate Retention"
        and row["revenue_at_risk"] > 0
    ):
        return "P1 - Immediate"

    elif row["retention_decision"] == "Prioritize Retention":
        return "P2 - High"

    elif row["retention_decision"] == "Consider Retention":
        return "P3 - Medium"

    elif row["retention_decision"] == "Monitor":
        return "P4 - Low"

    else:
        return "P5 - Do Not Spend"


df["roi_priority"] = df.apply(
    roi_priority,
    axis=1
)


# ---------------------------------------------------------
# 15. Business reason
# ---------------------------------------------------------

def roi_reason(row):

    if row["retention_decision"] == "Immediate Retention":
        return (
            "High expected value recovery relative to "
            "retention cost"
        )

    elif row["retention_decision"] == "Prioritize Retention":
        return (
            "Retention investment is expected to generate "
            "strong positive value"
        )

    elif row["retention_decision"] == "Consider Retention":
        return (
            "Potential value recovery is positive but "
            "requires selective targeting"
        )

    elif row["retention_decision"] == "Monitor":
        return (
            "Positive economics but relatively low "
            "retention priority"
        )

    else:
        return (
            "Expected recovery does not justify "
            "retention spending"
        )


df["roi_reason"] = df.apply(
    roi_reason,
    axis=1
)


# ---------------------------------------------------------
# 16. Final table
# ---------------------------------------------------------

result_df = df[
    [
        "customer_id",
        "estimated_clv",
        "churn_probability_ml",
        "revenue_at_risk",
        "revenue_risk_band",
        "revenue_priority",
        "recommended_action",
        "action_priority",
        "retention_strategy",
        "retention_cost",
        "expected_recovery_rate",
        "expected_recovered_value",
        "expected_net_benefit",
        "retention_roi",
        "retention_roi_percentage",
        "roi_category",
        "retention_decision",
        "roi_priority",
        "roi_reason"
    ]
].copy()


# ---------------------------------------------------------
# 17. Save to PostgreSQL
# ---------------------------------------------------------

print("Saving Retention ROI table...")

result_df.to_sql(
    "retention_roi",
    engine,
    if_exists="replace",
    index=False
)


# ---------------------------------------------------------
# 18. Business summary
# ---------------------------------------------------------

total_revenue_at_risk = result_df[
    "revenue_at_risk"
].sum()

total_retention_cost = result_df[
    "retention_cost"
].sum()

total_recovered_value = result_df[
    "expected_recovered_value"
].sum()

total_net_benefit = result_df[
    "expected_net_benefit"
].sum()

portfolio_roi = (
    total_net_benefit / total_retention_cost
    if total_retention_cost > 0
    else 0
)

customers_with_positive_roi = result_df[
    result_df["expected_net_benefit"] > 0
].shape[0]

customers_immediate = result_df[
    result_df["retention_decision"] == "Immediate Retention"
].shape[0]


# ---------------------------------------------------------
# 19. Print results
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 16 - RETENTION ROI SIMULATOR COMPLETED")
print("=" * 60)

print(
    f"Total customers: "
    f"{len(result_df):,}"
)

print(
    f"Total Revenue-at-Risk: "
    f"${total_revenue_at_risk:,.2f}"
)

print(
    f"Total retention cost: "
    f"${total_retention_cost:,.2f}"
)

print(
    f"Expected recovered value: "
    f"${total_recovered_value:,.2f}"
)

print(
    f"Expected net benefit: "
    f"${total_net_benefit:,.2f}"
)

print(
    f"Portfolio retention ROI: "
    f"{portfolio_roi:.2f}x"
)

print(
    f"Customers with positive ROI: "
    f"{customers_with_positive_roi:,}"
)

print(
    f"Immediate retention customers: "
    f"{customers_immediate:,}"
)

print()
print("ROI Category Distribution:")

print(
    result_df[
        "roi_category"
    ].value_counts()
)

print()
print("Retention Decision Distribution:")

print(
    result_df[
        "retention_decision"
    ].value_counts()
)

print()
print("Retention Strategy Distribution:")

print(
    result_df[
        "retention_strategy"
    ].value_counts()
)

print()
print("Top 10 Customers by Expected Net Benefit:")

top10 = result_df.sort_values(
    "expected_net_benefit",
    ascending=False
).head(10)

print(
    top10[
        [
            "customer_id",
            "revenue_at_risk",
            "retention_cost",
            "expected_recovered_value",
            "expected_net_benefit",
            "retention_roi",
            "retention_decision"
        ]
    ].to_string(index=False)
)

print()
print("Table created successfully: retention_roi")
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

print("Loading Revenue-at-Risk data...")


# ---------------------------------------------------------
# 2. Load churn predictions
# ---------------------------------------------------------

churn_query = """
SELECT
    customer_id,
    churn_probability_ml,
    predicted_churn
FROM churn_predictions
"""

churn_df = pd.read_sql(churn_query, engine)

print(f"Churn records loaded: {len(churn_df):,}")


# ---------------------------------------------------------
# 3. Load CLV
# ---------------------------------------------------------

clv_df = pd.read_sql(
    "SELECT * FROM customer_clv",
    engine
)

print(f"CLV records loaded: {len(clv_df):,}")

# Find CLV column
possible_clv_columns = [
    "estimated_clv",
    "customer_lifetime_value",
    "clv"
]

clv_column = None

for col in possible_clv_columns:
    if col in clv_df.columns:
        clv_column = col
        break

if clv_column is None:
    raise ValueError(
        f"Could not find CLV column. Available columns: "
        f"{clv_df.columns.tolist()}"
    )

clv_df = clv_df[
    ["customer_id", clv_column]
].copy()

clv_df.rename(
    columns={clv_column: "estimated_clv"},
    inplace=True
)


# ---------------------------------------------------------
# 4. Load customer risk
# ---------------------------------------------------------

risk_df = pd.read_sql(
    """
    SELECT
        customer_id,
        risk_score,
        risk_level
    FROM customer_risk
    """,
    engine
)

print(f"Risk records loaded: {len(risk_df):,}")


# ---------------------------------------------------------
# 5. Merge datasets
# ---------------------------------------------------------

df = churn_df.merge(
    clv_df,
    on="customer_id",
    how="left"
)

df = df.merge(
    risk_df,
    on="customer_id",
    how="left"
)

print(f"Customers after merge: {len(df):,}")


# ---------------------------------------------------------
# 6. Clean numeric fields
# ---------------------------------------------------------

df["churn_probability_ml"] = pd.to_numeric(
    df["churn_probability_ml"],
    errors="coerce"
).fillna(0)

df["estimated_clv"] = pd.to_numeric(
    df["estimated_clv"],
    errors="coerce"
).fillna(0)

df["risk_score"] = pd.to_numeric(
    df["risk_score"],
    errors="coerce"
).fillna(0)

df["churn_probability_ml"] = df[
    "churn_probability_ml"
].clip(0, 1)

df["estimated_clv"] = df[
    "estimated_clv"
].clip(lower=0)


# ---------------------------------------------------------
# 7. Calculate Revenue-at-Risk
# ---------------------------------------------------------

# Expected customer value that could be lost
df["revenue_at_risk"] = (
    df["estimated_clv"] *
    df["churn_probability_ml"]
)

# Customer value expected to remain
df["expected_retained_value"] = (
    df["estimated_clv"] *
    (1 - df["churn_probability_ml"])
)

# Percentage of CLV at risk
df["revenue_risk_percentage"] = (
    df["churn_probability_ml"] * 100
)


# ---------------------------------------------------------
# 8. Revenue-at-Risk bands
# ---------------------------------------------------------

def risk_band(probability):

    if probability >= 0.80:
        return "Critical"

    elif probability >= 0.60:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


df["revenue_risk_band"] = df[
    "churn_probability_ml"
].apply(risk_band)


# ---------------------------------------------------------
# 9. Revenue priority
# ---------------------------------------------------------

# Use the 75th percentile of Revenue-at-Risk
# to identify customers with particularly large
# potential value exposure.

risk_threshold = df[
    "revenue_at_risk"
].quantile(0.75)


def priority(row):

    if (
        row["churn_probability_ml"] >= 0.80
        and row["revenue_at_risk"] >= risk_threshold
    ):
        return "P1 - Critical Revenue Risk"

    elif row["churn_probability_ml"] >= 0.80:
        return "P2 - High Churn Risk"

    elif row["churn_probability_ml"] >= 0.60:
        return "P2 - High Revenue Risk"

    elif row["churn_probability_ml"] >= 0.40:
        return "P3 - Medium Revenue Risk"

    else:
        return "P4 - Low Revenue Risk"


df["revenue_priority"] = df.apply(
    priority,
    axis=1
)


# ---------------------------------------------------------
# 10. Revenue Risk Reason
# ---------------------------------------------------------

def reason(row):

    if row["revenue_risk_band"] == "Critical":
        return "Very high churn probability and significant customer value at risk"

    elif row["revenue_risk_band"] == "High":
        return "High churn probability creates significant potential value loss"

    elif row["revenue_risk_band"] == "Medium":
        return "Moderate churn probability requires monitoring"

    else:
        return "Low expected customer value exposure"


df["revenue_risk_reason"] = df.apply(
    reason,
    axis=1
)


# ---------------------------------------------------------
# 11. Select final columns
# ---------------------------------------------------------

result_df = df[
    [
        "customer_id",
        "estimated_clv",
        "churn_probability_ml",
        "predicted_churn",
        "risk_score",
        "risk_level",
        "revenue_at_risk",
        "expected_retained_value",
        "revenue_risk_percentage",
        "revenue_risk_band",
        "revenue_priority",
        "revenue_risk_reason"
    ]
].copy()


# ---------------------------------------------------------
# 12. Save to PostgreSQL
# ---------------------------------------------------------

print("Saving Revenue-at-Risk table...")

result_df.to_sql(
    "revenue_at_risk",
    engine,
    if_exists="replace",
    index=False
)


# ---------------------------------------------------------
# 13. Summary
# ---------------------------------------------------------

total_clv = result_df[
    "estimated_clv"
].sum()

total_revenue_at_risk = result_df[
    "revenue_at_risk"
].sum()

risk_percentage = (
    total_revenue_at_risk / total_clv * 100
    if total_clv > 0
    else 0
)

high_critical = result_df[
    result_df["revenue_risk_band"].isin(
        ["High", "Critical"]
    )
]

high_critical_revenue = high_critical[
    "revenue_at_risk"
].sum()


# ---------------------------------------------------------
# 14. Print results
# ---------------------------------------------------------

print()
print("=" * 60)
print("PHASE 15 - REVENUE-AT-RISK COMPLETED")
print("=" * 60)

print(f"Total customers: {len(result_df):,}")

print(
    f"Total estimated CLV: "
    f"${total_clv:,.2f}"
)

print(
    f"Total Revenue-at-Risk: "
    f"${total_revenue_at_risk:,.2f}"
)

print(
    f"Revenue-at-Risk percentage: "
    f"{risk_percentage:.2f}%"
)

print(
    f"High + Critical Revenue-at-Risk: "
    f"${high_critical_revenue:,.2f}"
)

print()
print("Revenue Risk Distribution:")

print(
    result_df[
        "revenue_risk_band"
    ].value_counts()
)

print()
print("Revenue Priority Distribution:")

print(
    result_df[
        "revenue_priority"
    ].value_counts()
)

print()
print("Top 10 Customers by Revenue-at-Risk:")

top10 = result_df.sort_values(
    "revenue_at_risk",
    ascending=False
).head(10)

print(
    top10[
        [
            "customer_id",
            "estimated_clv",
            "churn_probability_ml",
            "revenue_at_risk",
            "revenue_risk_band",
            "revenue_priority"
        ]
    ].to_string(index=False)
)

print()
print("Table created successfully: revenue_at_risk")
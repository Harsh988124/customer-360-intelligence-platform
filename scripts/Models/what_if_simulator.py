import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ============================================================
# PHASE 17 - WHAT-IF RETENTION SIMULATOR
# ============================================================

print("Loading What-If Simulation data...")

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Load Revenue-at-Risk data
# ------------------------------------------------------------

revenue_df = pd.read_sql(
    """
    SELECT
        customer_id,
        revenue_at_risk,
        revenue_risk_band

    FROM revenue_at_risk
    """,
    engine
)

print(f"Revenue-at-Risk records loaded: {len(revenue_df):,}")

# ------------------------------------------------------------
# Load Retention ROI data
# ------------------------------------------------------------

roi_df = pd.read_sql(
    """
    SELECT
        customer_id,
        retention_strategy,
        retention_cost,
        revenue_at_risk,
        expected_recovered_value,
        expected_net_benefit,
        retention_roi,
        retention_decision
    FROM retention_roi
    """,
    engine
)

print(f"Retention ROI records loaded: {len(roi_df):,}")

# ------------------------------------------------------------
# Merge datasets
# ------------------------------------------------------------

df = revenue_df.merge(
    roi_df,
    on="customer_id",
    how="left",
    suffixes=("_revenue", "_roi")
)

# Use one revenue-at-risk column
if "revenue_at_risk_revenue" in df.columns:
    df["revenue_at_risk"] = df["revenue_at_risk_revenue"]

print(f"Customers after merge: {len(df):,}")

# ------------------------------------------------------------
# Clean numeric fields
# ------------------------------------------------------------

df["revenue_at_risk"] = pd.to_numeric(
    df["revenue_at_risk"],
    errors="coerce"
).fillna(0)

df["retention_cost"] = pd.to_numeric(
    df["retention_cost"],
    errors="coerce"
).fillna(0)

# ------------------------------------------------------------
# What-if scenarios
# ------------------------------------------------------------

scenarios = [
    ("Conservative", 0.30),
    ("Moderate", 0.40),
    ("Base Case", 0.50),
    ("Optimistic", 0.60),
    ("Strong Retention", 0.70),
    ("Best Case", 0.80)
]

results = []

total_revenue_at_risk = df["revenue_at_risk"].sum()

# Customers considered targetable
targetable_df = df[
    df["revenue_at_risk"] > 0
].copy()

targetable_customers = len(targetable_df)

# ------------------------------------------------------------
# Run scenarios
# ------------------------------------------------------------

for scenario_name, recovery_rate in scenarios:

    # Expected recovered revenue
    expected_recovered_value = (
        targetable_df["revenue_at_risk"] *
        recovery_rate
    )

    total_recovered = expected_recovered_value.sum()

    # Retention cost
    total_retention_cost = targetable_df["retention_cost"].sum()

    # Net benefit
    net_benefit = (
        total_recovered -
        total_retention_cost
    )

    # ROI
    if total_retention_cost > 0:
        roi = net_benefit / total_retention_cost
    else:
        roi = 0

    # Revenue remaining at risk
    revenue_still_at_risk = (
        total_revenue_at_risk -
        total_recovered
    )

    results.append({
        "scenario_name": scenario_name,
        "recovery_rate": recovery_rate,
        "targetable_customers": targetable_customers,
        "total_revenue_at_risk": total_revenue_at_risk,
        "expected_recovered_value": total_recovered,
        "total_retention_cost": total_retention_cost,
        "expected_net_benefit": net_benefit,
        "retention_roi": roi,
        "revenue_still_at_risk": revenue_still_at_risk
    })

# ------------------------------------------------------------
# Create result dataframe
# ------------------------------------------------------------

simulation_df = pd.DataFrame(results)

# ------------------------------------------------------------
# Additional scenario comparison metrics
# ------------------------------------------------------------

base_recovered = simulation_df.loc[
    simulation_df["scenario_name"] == "Base Case",
    "expected_recovered_value"
].iloc[0]

simulation_df["additional_recovered_vs_base"] = (
    simulation_df["expected_recovered_value"] -
    base_recovered
)

# ------------------------------------------------------------
# Scenario recommendation
# ------------------------------------------------------------

def get_recommendation(row):

    if row["retention_roi"] >= 5:
        return "Highly Attractive"

    elif row["retention_roi"] >= 2:
        return "Strong Opportunity"

    elif row["retention_roi"] >= 1:
        return "Positive Opportunity"

    elif row["retention_roi"] >= 0:
        return "Marginal Opportunity"

    else:
        return "Not Recommended"


simulation_df["scenario_recommendation"] = (
    simulation_df.apply(
        get_recommendation,
        axis=1
    )
)

# ------------------------------------------------------------
# Round values
# ------------------------------------------------------------

numeric_columns = [
    "recovery_rate",
    "total_revenue_at_risk",
    "expected_recovered_value",
    "total_retention_cost",
    "expected_net_benefit",
    "retention_roi",
    "revenue_still_at_risk",
    "additional_recovered_vs_base"
]

simulation_df[numeric_columns] = (
    simulation_df[numeric_columns]
    .round(4)
)

# ------------------------------------------------------------
# Save to PostgreSQL
# ------------------------------------------------------------

print("Saving What-If Simulation table...")

simulation_df.to_sql(
    "what_if_simulation",
    engine,
    if_exists="replace",
    index=False
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 17 - WHAT-IF SIMULATOR COMPLETED")
print("=" * 70)

print(f"Total customers: {len(df):,}")
print(f"Targetable customers: {targetable_customers:,}")
print(f"Total Revenue-at-Risk: ${total_revenue_at_risk:,.2f}")

print()
print("SCENARIO RESULTS")
print("-" * 70)

for _, row in simulation_df.iterrows():

    print(
        f"{row['scenario_name']:20s} | "
        f"Recovery: {row['recovery_rate']:.0%} | "
        f"Recovered: ${row['expected_recovered_value']:,.2f} | "
        f"Net Benefit: ${row['expected_net_benefit']:,.2f} | "
        f"ROI: {row['retention_roi']:.2f}x"
    )

print()
print("BEST SCENARIO BY NET BENEFIT")
print("-" * 70)

best = simulation_df.loc[
    simulation_df["expected_net_benefit"].idxmax()
]

print(f"Scenario: {best['scenario_name']}")
print(f"Recovery Rate: {best['recovery_rate']:.0%}")
print(
    f"Expected Recovered Value: "
    f"${best['expected_recovered_value']:,.2f}"
)
print(
    f"Expected Net Benefit: "
    f"${best['expected_net_benefit']:,.2f}"
)
print(
    f"ROI: "
    f"{best['retention_roi']:.2f}x"
)

print()
print("Table created successfully: what_if_simulation")
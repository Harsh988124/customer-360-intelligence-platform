
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =========================================================
# CUSTOMER INTELLIGENCE PLATFORM
# PHASE 5 - EXPLORATORY DATA ANALYSIS
# =========================================================


# =========================================================
# 1. SETUP PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "charts"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. LOAD CUSTOMER 360 DATA
# =========================================================

print("\n" + "=" * 65)
print("CUSTOMER INTELLIGENCE PLATFORM")
print("PHASE 5 - EXPLORATORY DATA ANALYSIS")
print("=" * 65)


file_path = (
    PROCESSED_DIR
    / "customer_360.csv"
)

df = pd.read_csv(
    file_path
)


print(
    f"\nTotal Customers: "
    f"{len(df):,}"
)

print(
    f"Total Columns: "
    f"{len(df.columns)}"
)


# =========================================================
# 3. OVERALL CHURN ANALYSIS
# =========================================================

print("\n" + "=" * 65)
print("OVERALL CHURN ANALYSIS")
print("=" * 65)


total_customers = len(df)

churned_customers = df[
    "churned"
].sum()

churn_rate = (
    churned_customers
    / total_customers
    * 100
)


print(
    f"Total Customers: "
    f"{total_customers:,}"
)

print(
    f"Churned Customers: "
    f"{churned_customers:,}"
)

print(
    f"Overall Churn Rate: "
    f"{churn_rate:.2f}%"
)


# =========================================================
# 4. CHURN BY SUBSCRIPTION PLAN
# =========================================================

print("\n" + "=" * 65)
print("CHURN BY SUBSCRIPTION PLAN")
print("=" * 65)


plan_churn = (
    df
    .groupby("plan_name")
    .agg(
        customers=(
            "customer_id",
            "count"
        ),

        churned_customers=(
            "churned",
            "sum"
        ),

        churn_rate=(
            "churned",
            "mean"
        )
    )
    .reset_index()
)


plan_churn[
    "churn_rate"
] = (
    plan_churn[
        "churn_rate"
    ]
    * 100
).round(2)


print(
    plan_churn
)


# =========================================================
# 5. CHURN BY CONTRACT TYPE
# =========================================================

contract_churn = (
    df
    .groupby("contract_type")
    .agg(
        customers=(
            "customer_id",
            "count"
        ),

        churn_rate=(
            "churned",
            "mean"
        )
    )
    .reset_index()
)


contract_churn[
    "churn_rate"
] = (
    contract_churn[
        "churn_rate"
    ]
    * 100
).round(2)


print("\nCHURN BY CONTRACT TYPE")

print(
    contract_churn
)


# =========================================================
# 6. CHURN BY ENGAGEMENT
# =========================================================

engagement_churn = (
    df
    .groupby("engagement_segment")
    .agg(
        customers=(
            "customer_id",
            "count"
        ),

        churn_rate=(
            "churned",
            "mean"
        )
    )
    .reset_index()
)


engagement_churn[
    "churn_rate"
] = (
    engagement_churn[
        "churn_rate"
    ]
    * 100
).round(2)


print("\nCHURN BY ENGAGEMENT")

print(
    engagement_churn
)


# =========================================================
# 7. CHURN BY SUPPORT RISK
# =========================================================

support_churn = (
    df
    .groupby("support_risk")
    .agg(
        customers=(
            "customer_id",
            "count"
        ),

        churn_rate=(
            "churned",
            "mean"
        )
    )
    .reset_index()
)


support_churn[
    "churn_rate"
] = (
    support_churn[
        "churn_rate"
    ]
    * 100
).round(2)


print("\nCHURN BY SUPPORT RISK")

print(
    support_churn
)


# =========================================================
# 8. REVENUE BY CITY
# =========================================================

city_revenue = (
    df
    .groupby("city")
    .agg(
        customers=(
            "customer_id",
            "count"
        ),

        total_revenue=(
            "total_spent",
            "sum"
        ),

        average_customer_value=(
            "total_spent",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        by="total_revenue",
        ascending=False
    )
)


city_revenue[
    "total_revenue"
] = city_revenue[
    "total_revenue"
].round(2)


city_revenue[
    "average_customer_value"
] = city_revenue[
    "average_customer_value"
].round(2)


print("\nREVENUE BY CITY")

print(
    city_revenue
)


# =========================================================
# 9. CHURN BY AGE GROUP
# =========================================================

df["age_group"] = pd.cut(
    df["age"],
    bins=[
        17,
        25,
        35,
        45,
        55,
        100
    ],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]
)


age_churn = (
    df
    .groupby(
        "age_group",
        observed=False
    )
    .agg(
        customers=(
            "customer_id",
            "count"
        ),

        churn_rate=(
            "churned",
            "mean"
        )
    )
    .reset_index()
)


age_churn[
    "churn_rate"
] = (
    age_churn[
        "churn_rate"
    ]
    * 100
).round(2)


print("\nCHURN BY AGE GROUP")

print(
    age_churn
)


# =========================================================
# 10. CREATE CHARTS
# =========================================================

print("\nCreating charts...")


sns.set_theme(
    style="whitegrid"
)


# ---------------------------------------------------------
# Chart 1 - Churn Distribution
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.countplot(
    data=df,
    x="churned"
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Churned"
)

plt.ylabel(
    "Number of Customers"
)

plt.savefig(
    OUTPUT_DIR
    / "churn_distribution.png",
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Chart 2 - Churn Rate by Plan
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.barplot(
    data=plan_churn,
    x="plan_name",
    y="churn_rate"
)

plt.title(
    "Churn Rate by Subscription Plan"
)

plt.xlabel(
    "Subscription Plan"
)

plt.ylabel(
    "Churn Rate (%)"
)

plt.savefig(
    OUTPUT_DIR
    / "churn_by_plan.png",
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Chart 3 - Churn Rate by Contract
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.barplot(
    data=contract_churn,
    x="contract_type",
    y="churn_rate"
)

plt.title(
    "Churn Rate by Contract Type"
)

plt.xlabel(
    "Contract Type"
)

plt.ylabel(
    "Churn Rate (%)"
)

plt.savefig(
    OUTPUT_DIR
    / "churn_by_contract.png",
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Chart 4 - Churn by Engagement
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.barplot(
    data=engagement_churn,
    x="engagement_segment",
    y="churn_rate"
)

plt.title(
    "Churn Rate by Customer Engagement"
)

plt.xlabel(
    "Engagement Level"
)

plt.ylabel(
    "Churn Rate (%)"
)

plt.savefig(
    OUTPUT_DIR
    / "churn_by_engagement.png",
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Chart 5 - Churn by Support Risk
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.barplot(
    data=support_churn,
    x="support_risk",
    y="churn_rate"
)

plt.title(
    "Churn Rate by Support Risk"
)

plt.xlabel(
    "Support Risk"
)

plt.ylabel(
    "Churn Rate (%)"
)

plt.savefig(
    OUTPUT_DIR
    / "churn_by_support_risk.png",
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Chart 6 - Revenue by City
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

sns.barplot(
    data=city_revenue,
    x="city",
    y="total_revenue"
)

plt.title(
    "Total Revenue by City"
)

plt.xlabel(
    "City"
)

plt.ylabel(
    "Total Revenue"
)

plt.xticks(
    rotation=45
)

plt.savefig(
    OUTPUT_DIR
    / "revenue_by_city.png",
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Chart 7 - Churn by Age Group
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

sns.barplot(
    data=age_churn,
    x="age_group",
    y="churn_rate"
)

plt.title(
    "Churn Rate by Age Group"
)

plt.xlabel(
    "Age Group"
)

plt.ylabel(
    "Churn Rate (%)"
)

plt.savefig(
    OUTPUT_DIR
    / "churn_by_age_group.png",
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 11. CREATE SUMMARY REPORT
# =========================================================

summary = f"""
CUSTOMER INTELLIGENCE PLATFORM
PHASE 5 - EXPLORATORY DATA ANALYSIS

Total Customers: {total_customers}
Churned Customers: {churned_customers}
Overall Churn Rate: {churn_rate:.2f}%

KEY ANALYSIS COMPLETED:
- Overall churn analysis
- Churn by subscription plan
- Churn by contract type
- Churn by engagement
- Churn by support risk
- Revenue by city
- Churn by age group

"""

report_path = (
    REPORT_DIR
    / "eda_summary.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary
    )


# =========================================================
# 12. SAVE ANALYTICAL TABLES
# =========================================================

plan_churn.to_csv(
    REPORT_DIR
    / "plan_churn_analysis.csv",
    index=False
)

contract_churn.to_csv(
    REPORT_DIR
    / "contract_churn_analysis.csv",
    index=False
)

engagement_churn.to_csv(
    REPORT_DIR
    / "engagement_churn_analysis.csv",
    index=False
)

support_churn.to_csv(
    REPORT_DIR
    / "support_churn_analysis.csv",
    index=False
)

city_revenue.to_csv(
    REPORT_DIR
    / "city_revenue_analysis.csv",
    index=False
)

age_churn.to_csv(
    REPORT_DIR
    / "age_churn_analysis.csv",
    index=False
)


# =========================================================
# 13. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("PHASE 5 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"\nCharts saved in:"
    f"\n{OUTPUT_DIR}"
)

print(
    f"\nReports saved in:"
    f"\n{REPORT_DIR}"
)

print(
    "\nExploratory Data Analysis "
    "completed successfully!"
)

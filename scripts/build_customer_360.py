
import pandas as pd
from pathlib import Path


# =========================================================
# CUSTOMER INTELLIGENCE PLATFORM
# CUSTOMER 360 DATA ENGINEERING PIPELINE
# =========================================================


# =========================================================
# 1. SETUP PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "customer_360.csv"
)


print("\n" + "=" * 65)
print("CUSTOMER 360 DATA ENGINEERING PIPELINE")
print("=" * 65)


# =========================================================
# 2. LOAD CLEANED DATASETS
# =========================================================

print("\nLoading processed datasets...")


customers = pd.read_csv(
    PROCESSED_DIR
    / "customers_clean.csv"
)

subscriptions = pd.read_csv(
    PROCESSED_DIR
    / "subscriptions_clean.csv"
)

transactions = pd.read_csv(
    PROCESSED_DIR
    / "transactions_clean.csv"
)

usage = pd.read_csv(
    PROCESSED_DIR
    / "usage_data_clean.csv"
)

tickets = pd.read_csv(
    PROCESSED_DIR
    / "support_tickets_clean.csv"
)


print(
    f"Customers: {len(customers):,}"
)

print(
    f"Subscriptions: {len(subscriptions):,}"
)

print(
    f"Transactions: {len(transactions):,}"
)

print(
    f"Usage Records: {len(usage):,}"
)

print(
    f"Support Tickets: {len(tickets):,}"
)


# =========================================================
# 3. DATA TYPE CONVERSION
# =========================================================

print("\nConverting date columns...")


customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"]
)

usage["usage_date"] = pd.to_datetime(
    usage["usage_date"]
)

tickets["ticket_date"] = pd.to_datetime(
    tickets["ticket_date"]
)


# =========================================================
# 4. TRANSACTION FEATURES
# =========================================================

print("\nCreating transaction features...")


transaction_features = (
    transactions
    .groupby("customer_id")
    .agg(
        total_transactions=(
            "transaction_id",
            "count"
        ),

        total_spent=(
            "amount",
            "sum"
        ),

        average_transaction_amount=(
            "amount",
            "mean"
        ),

        successful_transactions=(
            "transaction_status",
            lambda x: (
                x == "Success"
            ).sum()
        ),

        failed_transactions=(
            "transaction_status",
            lambda x: (
                x == "Failed"
            ).sum()
        )
    )
    .reset_index()
)


transaction_features[
    "average_transaction_amount"
] = transaction_features[
    "average_transaction_amount"
].round(2)


# =========================================================
# 5. USAGE FEATURES
# =========================================================

print("\nCreating usage features...")


usage_features = (
    usage
    .groupby("customer_id")
    .agg(
        usage_days=(
            "usage_date",
            "nunique"
        ),

        total_logins=(
            "login_count",
            "sum"
        ),

        average_logins=(
            "login_count",
            "mean"
        ),

        total_session_minutes=(
            "session_minutes",
            "sum"
        ),

        average_session_minutes=(
            "session_minutes",
            "mean"
        ),

        total_features_used=(
            "features_used",
            "sum"
        ),

        average_features_used=(
            "features_used",
            "mean"
        )
    )
    .reset_index()
)


usage_features[
    "average_logins"
] = usage_features[
    "average_logins"
].round(2)


usage_features[
    "average_session_minutes"
] = usage_features[
    "average_session_minutes"
].round(2)


usage_features[
    "average_features_used"
] = usage_features[
    "average_features_used"
].round(2)


# =========================================================
# 6. SUPPORT FEATURES
# =========================================================

print("\nCreating support features...")


support_features = (
    tickets
    .groupby("customer_id")
    .agg(
        total_tickets=(
            "ticket_id",
            "count"
        ),

        average_resolution_hours=(
            "resolution_hours",
            "mean"
        ),

        negative_tickets=(
            "customer_sentiment",
            lambda x: (
                x == "Negative"
            ).sum()
        ),

        positive_tickets=(
            "customer_sentiment",
            lambda x: (
                x == "Positive"
            ).sum()
        ),

        open_tickets=(
            "ticket_status",
            lambda x: (
                x == "Open"
            ).sum()
        )
    )
    .reset_index()
)


support_features[
    "average_resolution_hours"
] = support_features[
    "average_resolution_hours"
].round(2)


# =========================================================
# 7. SUBSCRIPTION FEATURES
# =========================================================

print("\nPreparing subscription features...")


subscription_features = subscriptions[[
    "customer_id",
    "plan_name",
    "monthly_price",
    "contract_type",
    "auto_renew"
]].copy()


# =========================================================
# 8. BUILD CUSTOMER 360 TABLE
# =========================================================

print("\nBuilding Customer 360 table...")


customer_360 = customers.copy()


# Merge subscription information
customer_360 = customer_360.merge(
    subscription_features,
    on="customer_id",
    how="left"
)


# Merge transaction features
customer_360 = customer_360.merge(
    transaction_features,
    on="customer_id",
    how="left"
)


# Merge usage features
customer_360 = customer_360.merge(
    usage_features,
    on="customer_id",
    how="left"
)


# Merge support features
customer_360 = customer_360.merge(
    support_features,
    on="customer_id",
    how="left"
)


# =========================================================
# 9. HANDLE MISSING VALUES
# =========================================================

print("\nHandling missing values...")


numeric_columns = customer_360.select_dtypes(
    include="number"
).columns


customer_360[numeric_columns] = (
    customer_360[numeric_columns]
    .fillna(0)
)


categorical_columns = customer_360.select_dtypes(
    include="object"
).columns


customer_360[categorical_columns] = (
    customer_360[categorical_columns]
    .fillna("Unknown")
)


# =========================================================
# 10. CREATE ADDITIONAL BUSINESS FEATURES
# =========================================================

print(
    "\nCreating additional "
    "customer intelligence features..."
)


# ---------------------------------------------------------
# CUSTOMER TENURE
# ---------------------------------------------------------

current_date = pd.Timestamp.today()


customer_360[
    "customer_tenure_days"
] = (
    current_date
    - customer_360[
        "signup_date"
    ]
).dt.days


customer_360[
    "customer_tenure_months"
] = (
    customer_360[
        "customer_tenure_days"
    ]
    / 30
).round(1)


# ---------------------------------------------------------
# CUSTOMER VALUE SEGMENT
# ---------------------------------------------------------

customer_360[
    "customer_value_segment"
] = pd.cut(
    customer_360[
        "total_spent"
    ],
    bins=[
        -1,
        5000,
        15000,
        float("inf")
    ],
    labels=[
        "Low Value",
        "Medium Value",
        "High Value"
    ]
)


# ---------------------------------------------------------
# ENGAGEMENT SEGMENT
# ---------------------------------------------------------

customer_360[
    "engagement_segment"
] = pd.cut(
    customer_360[
        "average_logins"
    ],
    bins=[
        -1,
        1,
        3,
        float("inf")
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)


# ---------------------------------------------------------
# SUPPORT RISK
# ---------------------------------------------------------

customer_360[
    "support_risk"
] = "Low"


customer_360.loc[
    customer_360[
        "negative_tickets"
    ] >= 2,
    "support_risk"
] = "High"


customer_360.loc[
    (
        customer_360[
            "negative_tickets"
        ] == 1
    )
    &
    (
        customer_360[
            "support_risk"
        ] != "High"
    ),
    "support_risk"
] = "Medium"


# ---------------------------------------------------------
# CUSTOMER VALUE SCORE
# ---------------------------------------------------------

customer_360[
    "customer_value_score"
] = (

    customer_360[
        "total_spent"
    ].rank(
        pct=True
    )

    * 100
).round(2)


# =========================================================
# 11. FINAL COLUMN CLEANUP
# =========================================================

customer_360[
    "customer_value_segment"
] = customer_360[
    "customer_value_segment"
].astype(str)


customer_360[
    "engagement_segment"
] = customer_360[
    "engagement_segment"
].astype(str)


# =========================================================
# 12. SAVE CUSTOMER 360
# =========================================================

customer_360.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 13. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("CUSTOMER 360 CREATED SUCCESSFULLY")
print("=" * 65)


print(
    f"Total Customers: "
    f"{len(customer_360):,}"
)


print(
    f"Total Columns: "
    f"{len(customer_360.columns)}"
)


print(
    f"\nOutput File:\n"
    f"{OUTPUT_FILE}"
)


print("\nSample Data:")


print(
    customer_360.head()
)


print(
    "\nCustomer 360 pipeline "
    "completed successfully!"
)


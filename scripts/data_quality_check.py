
import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CUSTOMER INTELLIGENCE PLATFORM
# DATA QUALITY PIPELINE - VERSION 2
# =========================================================


# =========================================================
# 1. SETUP PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. DATASET CONFIGURATION
# =========================================================

DATASETS = {
    "customers": {
        "file": "customers.csv",
        "required_columns": [
            "customer_id",
            "first_name",
            "last_name",
            "age",
            "gender",
            "city",
            "country",
            "signup_date",
            "customer_status",
            "churn_probability",
            "churned"
        ],
        "primary_key": "customer_id"
    },

    "subscriptions": {
        "file": "subscriptions.csv",
        "required_columns": [
            "subscription_id",
            "customer_id",
            "plan_name",
            "monthly_price",
            "contract_type",
            "auto_renew"
        ],
        "primary_key": "subscription_id"
    },

    "transactions": {
        "file": "transactions.csv",
        "required_columns": [
            "transaction_id",
            "customer_id",
            "transaction_date",
            "amount",
            "payment_method",
            "transaction_status",
            "product_category"
        ],
        "primary_key": "transaction_id"
    },

    "usage_data": {
        "file": "usage_data.csv",
        "required_columns": [
            "usage_id",
            "customer_id",
            "usage_date",
            "login_count",
            "session_minutes",
            "features_used",
            "device_type"
        ],
        "primary_key": "usage_id"
    },

    "support_tickets": {
        "file": "support_tickets.csv",
        "required_columns": [
            "ticket_id",
            "customer_id",
            "ticket_date",
            "issue_category",
            "priority",
            "resolution_hours",
            "ticket_status",
            "customer_sentiment"
        ],
        "primary_key": "ticket_id"
    }
}


# =========================================================
# 3. DATE COLUMN CONFIGURATION
# =========================================================

DATE_COLUMNS = {
    "customers": [
        "signup_date"
    ],

    "subscriptions": [],

    "transactions": [
        "transaction_date"
    ],

    "usage_data": [
        "usage_date"
    ],

    "support_tickets": [
        "ticket_date"
    ]
}


# =========================================================
# 4. ALLOWED CATEGORY RULES
# =========================================================

CATEGORY_RULES = {

    "customers": {

        "gender": [
            "Male",
            "Female",
            "Other"
        ],

        "customer_status": [
            "Active",
            "Churned",
            "Inactive"
        ]
    },


    "subscriptions": {

        "plan_name": [
            "Basic",
            "Standard",
            "Premium"
        ],

        "contract_type": [
            "Monthly",
            "Yearly"
        ],

        "auto_renew": [
            "Yes",
            "No"
        ]
    },


    "transactions": {

        "transaction_status": [
            "Success",
            "Failed"
        ],

        "payment_method": [
            "UPI",
            "Credit Card",
            "Debit Card",
            "Net Banking"
        ],

        "product_category": [
            "Subscription",
            "Add-on",
            "Premium Service"
        ]
    },


    "usage_data": {

        "device_type": [
            "Mobile",
            "Web",
            "Desktop"
        ]
    },


    "support_tickets": {

        "issue_category": [
            "Billing",
            "Technical",
            "Login",
            "Subscription",
            "Payment"
        ],

        "priority": [
            "Low",
            "Medium",
            "High"
        ],

        "ticket_status": [
            "Resolved",
            "Open"
        ],

        "customer_sentiment": [
            "Positive",
            "Neutral",
            "Negative"
        ]
    }
}


# =========================================================
# 5. PRIMARY KEY VALIDATION
# =========================================================

def validate_primary_key(df, column_name):

    if column_name not in df.columns:

        return {
            "status": "FAIL",
            "nulls": 0,
            "duplicates": 0,
            "error": "Primary key column missing"
        }

    null_count = int(
        df[column_name].isnull().sum()
    )

    duplicate_count = int(
        df[column_name].duplicated().sum()
    )

    if null_count == 0 and duplicate_count == 0:

        status = "PASS"

    else:

        status = "FAIL"

    return {
        "status": status,
        "nulls": null_count,
        "duplicates": duplicate_count,
        "error": None
    }


# =========================================================
# 6. FOREIGN KEY VALIDATION
# =========================================================

def validate_foreign_key(
    child_df,
    parent_df,
    child_column,
    parent_column
):

    if (
        child_column not in child_df.columns
        or parent_column not in parent_df.columns
    ):

        return {
            "status": "FAIL",
            "invalid_records": 0,
            "error": "Foreign key column missing"
        }

    invalid_records = child_df[
        ~child_df[child_column].isin(
            parent_df[parent_column]
        )
    ]

    invalid_count = len(
        invalid_records
    )

    if invalid_count == 0:

        status = "PASS"

    else:

        status = "FAIL"

    return {
        "status": status,
        "invalid_records": int(
            invalid_count
        ),
        "error": None
    }


# =========================================================
# 7. DATE VALIDATION
# =========================================================

def validate_dates(
    df,
    date_columns
):

    results = {}

    for column in date_columns:

        if column not in df.columns:

            results[column] = {
                "status": "FAIL",
                "invalid_dates": 0,
                "future_dates": 0,
                "error": "Date column missing"
            }

            continue

        converted_dates = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        invalid_dates = int(
            converted_dates.isna().sum()
        )

        future_dates = int(
            (
                converted_dates
                > pd.Timestamp.today()
            ).sum()
        )

        if (
            invalid_dates == 0
            and future_dates == 0
        ):

            status = "PASS"

        else:

            status = "FAIL"

        results[column] = {
            "status": status,
            "invalid_dates":
                invalid_dates,
            "future_dates":
                future_dates,
            "error": None
        }

    return results


# =========================================================
# 8. CATEGORY VALIDATION
# =========================================================

def validate_categories(
    df,
    column,
    allowed_values
):

    if column not in df.columns:

        return {
            "status": "FAIL",
            "invalid_values": 0,
            "error": "Category column missing"
        }

    invalid_records = df[
        ~df[column].isin(
            allowed_values
        )
        &
        df[column].notna()
    ]

    invalid_count = len(
        invalid_records
    )

    if invalid_count == 0:

        status = "PASS"

    else:

        status = "FAIL"

    return {
        "status": status,
        "invalid_values":
            int(invalid_count),
        "error": None
    }


# =========================================================
# 9. BUSINESS RULE VALIDATION
# =========================================================

def validate_customers(df):

    issues = 0

    invalid_age = len(
        df[
            (df["age"] < 18)
            |
            (df["age"] > 100)
        ]
    )

    invalid_churn = len(
        df[
            ~df["churned"].isin([0, 1])
        ]
    )

    invalid_probability = len(
        df[
            (df["churn_probability"] < 0)
            |
            (df["churn_probability"] > 1)
        ]
    )

    issues += invalid_age
    issues += invalid_churn
    issues += invalid_probability

    return {
        "invalid_age": invalid_age,
        "invalid_churn": invalid_churn,
        "invalid_probability":
            invalid_probability,
        "total_issues": issues
    }


def validate_subscriptions(df):

    invalid_price = len(
        df[
            df["monthly_price"] <= 0
        ]
    )

    return {
        "invalid_price":
            invalid_price,
        "total_issues":
            invalid_price
    }


def validate_transactions(df):

    invalid_amount = len(
        df[
            df["amount"] <= 0
        ]
    )

    return {
        "invalid_amount":
            invalid_amount,
        "total_issues":
            invalid_amount
    }


def validate_usage(df):

    invalid_logins = len(
        df[
            df["login_count"] < 0
        ]
    )

    invalid_session = len(
        df[
            df["session_minutes"] < 0
        ]
    )

    invalid_features = len(
        df[
            df["features_used"] < 0
        ]
    )

    total_issues = (
        invalid_logins
        + invalid_session
        + invalid_features
    )

    return {
        "invalid_logins":
            invalid_logins,

        "invalid_session":
            invalid_session,

        "invalid_features":
            invalid_features,

        "total_issues":
            total_issues
    }


def validate_tickets(df):

    invalid_resolution = len(
        df[
            df["resolution_hours"] < 0
        ]
    )

    return {
        "invalid_resolution":
            invalid_resolution,

        "total_issues":
            invalid_resolution
    }


# =========================================================
# 10. DATA QUALITY SCORE
# =========================================================

def calculate_quality_score(
    df,
    total_invalid_records
):

    total_rows = len(df)

    if total_rows == 0:

        return 0

    total_cells = (
        df.shape[0]
        * df.shape[1]
    )

    # Missing value percentage
    missing_values = (
        df.isnull().sum().sum()
    )

    missing_percentage = (
        missing_values
        / total_cells
        * 100
    )

    # Duplicate percentage
    duplicate_records = (
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_records
        / total_rows
        * 100
    )

    # Invalid record percentage
    invalid_percentage = min(
        (
            total_invalid_records
            / total_rows
            * 100
        ),
        100
    )

    # Final score
    score = (
        100
        - missing_percentage
        - duplicate_percentage
        - invalid_percentage
    )

    score = max(
        0,
        round(score, 2)
    )

    return score


# =========================================================
# 11. LOAD DATASETS
# =========================================================

print("\n")
print("=" * 65)
print("CUSTOMER INTELLIGENCE PLATFORM")
print("DATA QUALITY PIPELINE - VERSION 2")
print("=" * 65)


dataframes = {}

for dataset_name, config in DATASETS.items():

    file_path = (
        RAW_DIR
        / config["file"]
    )

    print(
        f"\nLoading: "
        f"{file_path.name}"
    )

    try:

        df = pd.read_csv(
            file_path
        )

        dataframes[
            dataset_name
        ] = df

        print(
            f"✅ Loaded "
            f"{len(df):,} records"
        )

    except FileNotFoundError:

        print(
            f"❌ File not found: "
            f"{file_path}"
        )

        raise


# =========================================================
# 12. RUN ALL VALIDATIONS
# =========================================================

quality_results = []


print("\n")
print("=" * 65)
print("RUNNING DATA QUALITY VALIDATIONS")
print("=" * 65)


for dataset_name, config in DATASETS.items():

    df = dataframes[
        dataset_name
    ]

    print("\n")
    print("-" * 65)
    print(
        f"DATASET: "
        f"{dataset_name.upper()}"
    )
    print("-" * 65)


    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )


    # -----------------------------------------------------
    # REQUIRED COLUMN CHECK
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in config[
            "required_columns"
        ]
        if column not in df.columns
    ]

    if len(missing_columns) == 0:

        print(
            "Required Columns: PASS"
        )

        required_columns_status = "PASS"

    else:

        print(
            f"Required Columns: FAIL "
            f"{missing_columns}"
        )

        required_columns_status = "FAIL"


    # -----------------------------------------------------
    # MISSING VALUE CHECK
    # -----------------------------------------------------

    missing_values = int(
        df.isnull().sum().sum()
    )

    if missing_values == 0:

        print(
            "Missing Values: PASS"
        )

        missing_status = "PASS"

    else:

        print(
            f"Missing Values: FAIL "
            f"({missing_values})"
        )

        missing_status = "FAIL"


    # -----------------------------------------------------
    # DUPLICATE CHECK
    # -----------------------------------------------------

    duplicate_records = int(
        df.duplicated().sum()
    )

    if duplicate_records == 0:

        print(
            "Duplicate Records: PASS"
        )

        duplicate_status = "PASS"

    else:

        print(
            f"Duplicate Records: FAIL "
            f"({duplicate_records})"
        )

        duplicate_status = "FAIL"


    # -----------------------------------------------------
    # PRIMARY KEY CHECK
    # -----------------------------------------------------

    primary_key = config[
        "primary_key"
    ]

    primary_key_result = (
        validate_primary_key(
            df,
            primary_key
        )
    )

    print(
        f"Primary Key ({primary_key}): "
        f"{primary_key_result['status']}"
    )

    print(
        f"  Nulls: "
        f"{primary_key_result['nulls']}"
    )

    print(
        f"  Duplicates: "
        f"{primary_key_result['duplicates']}"
    )


    # -----------------------------------------------------
    # DATE VALIDATION
    # -----------------------------------------------------

    date_results = validate_dates(
        df,
        DATE_COLUMNS.get(
            dataset_name,
            []
        )
    )

    total_invalid_dates = 0
    total_future_dates = 0

    for column, result in (
        date_results.items()
    ):

        print(
            f"Date Check ({column}): "
            f"{result['status']}"
        )

        print(
            f"  Invalid Dates: "
            f"{result['invalid_dates']}"
        )

        print(
            f"  Future Dates: "
            f"{result['future_dates']}"
        )

        total_invalid_dates += (
            result[
                "invalid_dates"
            ]
        )

        total_future_dates += (
            result[
                "future_dates"
            ]
        )


    # -----------------------------------------------------
    # CATEGORY VALIDATION
    # -----------------------------------------------------

    category_results = {}

    total_invalid_categories = 0

    dataset_rules = (
        CATEGORY_RULES.get(
            dataset_name,
            {}
        )
    )

    for column, allowed_values in (
        dataset_rules.items()
    ):

        result = validate_categories(
            df,
            column,
            allowed_values
        )

        category_results[
            column
        ] = result

        print(
            f"Category ({column}): "
            f"{result['status']}"
        )

        print(
            f"  Invalid Values: "
            f"{result['invalid_values']}"
        )

        total_invalid_categories += (
            result[
                "invalid_values"
            ]
        )


    # -----------------------------------------------------
    # BUSINESS RULE VALIDATION
    # -----------------------------------------------------

    if dataset_name == "customers":

        business_result = (
            validate_customers(df)
        )

    elif dataset_name == "subscriptions":

        business_result = (
            validate_subscriptions(df)
        )

    elif dataset_name == "transactions":

        business_result = (
            validate_transactions(df)
        )

    elif dataset_name == "usage_data":

        business_result = (
            validate_usage(df)
        )

    elif dataset_name == "support_tickets":

        business_result = (
            validate_tickets(df)
        )


    total_business_issues = (
        business_result[
            "total_issues"
        ]
    )


    if total_business_issues == 0:

        business_status = "PASS"

    else:

        business_status = "FAIL"


    print(
        f"Business Rules: "
        f"{business_status}"
    )

    print(
        f"  Total Issues: "
        f"{total_business_issues}"
    )


    # =====================================================
    # 13. FOREIGN KEY VALIDATION
    # =====================================================

    foreign_key_invalid = 0

    if dataset_name != "customers":

        fk_result = (
            validate_foreign_key(
                df,
                dataframes["customers"],
                "customer_id",
                "customer_id"
            )
        )

        foreign_key_invalid = (
            fk_result[
                "invalid_records"
            ]
        )

        print(
            "Foreign Key "
            f"(customer_id): "
            f"{fk_result['status']}"
        )

        print(
            f"  Invalid Customer IDs: "
            f"{foreign_key_invalid}"
        )

        foreign_key_status = (
            fk_result["status"]
        )

    else:

        foreign_key_status = "N/A"

        print(
            "Foreign Key: N/A "
            "(Parent Table)"
        )


    # =====================================================
    # 14. CALCULATE TOTAL INVALID RECORDS
    # =====================================================

    total_invalid_records = (

        primary_key_result[
            "nulls"
        ]

        + primary_key_result[
            "duplicates"
        ]

        + total_invalid_dates

        + total_future_dates

        + total_invalid_categories

        + total_business_issues

        + foreign_key_invalid
    )


    # =====================================================
    # 15. CALCULATE DATA QUALITY SCORE
    # =====================================================

    quality_score = (
        calculate_quality_score(
            df,
            total_invalid_records
        )
    )


    # =====================================================
    # 16. FINAL DATASET STATUS
    # =====================================================

    all_checks = [

        required_columns_status,
        missing_status,
        duplicate_status,

        primary_key_result[
            "status"
        ],

        business_status
    ]


    # Add date validation statuses
    for result in (
        date_results.values()
    ):

        all_checks.append(
            result["status"]
        )


    # Add category validation statuses
    for result in (
        category_results.values()
    ):

        all_checks.append(
            result["status"]
        )


    # Add foreign key validation
    if foreign_key_status != "N/A":

        all_checks.append(
            foreign_key_status
        )


    if "FAIL" in all_checks:

        final_status = "WARNING"

    else:

        final_status = "PASS"


    print(
        f"\nFINAL STATUS: "
        f"{final_status}"
    )

    print(
        f"DATA QUALITY SCORE: "
        f"{quality_score}%"
    )


    # =====================================================
    # 17. SAVE RESULTS
    # =====================================================

    quality_results.append({

        "dataset":
            dataset_name,

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "missing_values":
            missing_values,

        "duplicate_records":
            duplicate_records,

        "primary_key_status":
            primary_key_result[
                "status"
            ],

        "primary_key_nulls":
            primary_key_result[
                "nulls"
            ],

        "primary_key_duplicates":
            primary_key_result[
                "duplicates"
            ],

        "invalid_dates":
            total_invalid_dates,

        "future_dates":
            total_future_dates,

        "invalid_category_values":
            total_invalid_categories,

        "business_rule_issues":
            total_business_issues,

        "foreign_key_invalid":
            foreign_key_invalid,

        "foreign_key_status":
            foreign_key_status,

        "quality_score":
            quality_score,

        "final_status":
            final_status
    })


# =========================================================
# 18. CREATE DATA QUALITY REPORT
# =========================================================

quality_report = pd.DataFrame(
    quality_results
)


report_path = (
    PROCESSED_DIR
    / "data_quality_report.csv"
)


quality_report.to_csv(
    report_path,
    index=False
)


print("\n")
print("=" * 65)
print("DATA QUALITY REPORT")
print("=" * 65)

print(
    quality_report.to_string(
        index=False
    )
)


print(
    f"\nReport saved to:\n"
    f"{report_path}"
)


# =========================================================
# 19. SAVE CLEANED DATA
# =========================================================

print("\n")
print("=" * 65)
print("SAVING PROCESSED DATASETS")
print("=" * 65)


for dataset_name, df in (
    dataframes.items()
):

    # Remove exact duplicate rows
    df_clean = df.drop_duplicates()

    output_path = (
        PROCESSED_DIR
        / f"{dataset_name}_clean.csv"
    )

    df_clean.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: "
        f"{output_path.name}"
    )


# =========================================================
# 20. FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 65)
print("PHASE 3 COMPLETED")
print("=" * 65)


average_score = round(
    quality_report[
        "quality_score"
    ].mean(),
    2
)


print(
    f"Overall Average "
    f"Data Quality Score: "
    f"{average_score}%"
)


print(
    f"\nQuality Report:"
    f"\n{report_path}"
)


print(
    "\nProcessed files saved in:"
)

print(
    PROCESSED_DIR
)


print("\nPipeline completed successfully!")

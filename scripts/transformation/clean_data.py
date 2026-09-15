"""
Data Cleaning Module
Customer 360 Intelligence & Retention Platform

Performs reusable data cleaning operations using PySpark.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lower, trim, to_date


def clean_customers(df: DataFrame) -> DataFrame:
    """
    Clean customer data.
    """

    df = (
        df
        .dropDuplicates(["customer_id"])
        .withColumn("customer_id", trim(col("customer_id")))
        .withColumn("first_name", trim(col("first_name")))
        .withColumn("last_name", trim(col("last_name")))
        .withColumn("gender", trim(col("gender")))
        .withColumn("city", trim(col("city")))
        .withColumn("country", trim(col("country")))
        .withColumn("signup_date", to_date(col("signup_date")))
    )

    df = df.filter(
        col("customer_id").isNotNull()
    )

    df = df.filter(
        (col("age") >= 18) &
        (col("age") <= 100)
    )

    return df


def clean_subscriptions(df: DataFrame) -> DataFrame:
    """
    Clean subscription data.
    """

    df = (
        df
        .dropDuplicates(["subscription_id"])
        .withColumn(
            "subscription_id",
            trim(col("subscription_id"))
        )
        .withColumn(
            "customer_id",
            trim(col("customer_id"))
        )
        .withColumn(
            "plan_name",
            trim(col("plan_name"))
        )
        .withColumn(
            "contract_type",
            trim(col("contract_type"))
        )
        .withColumn(
            "auto_renew",
            trim(col("auto_renew"))
        )
    )

    df = df.filter(
        col("customer_id").isNotNull()
    )

    return df


def clean_usage(df: DataFrame) -> DataFrame:
    """
    Clean usage data.
    """

    df = (
        df
        .dropDuplicates(["usage_id"])
        .withColumn(
            "usage_id",
            trim(col("usage_id"))
        )
        .withColumn(
            "customer_id",
            trim(col("customer_id"))
        )
        .withColumn(
            "usage_date",
            to_date(col("usage_date"))
        )
        .withColumn(
            "device_type",
            trim(col("device_type"))
        )
    )

    df = df.filter(
        col("customer_id").isNotNull()
    )

    df = df.filter(
        (col("login_count") >= 0) &
        (col("session_minutes") >= 0) &
        (col("features_used") >= 0)
    )

    return df


def clean_support(df: DataFrame) -> DataFrame:
    """
    Clean support ticket data.
    """

    df = (
        df
        .dropDuplicates(["ticket_id"])
        .withColumn(
            "ticket_id",
            trim(col("ticket_id"))
        )
        .withColumn(
            "customer_id",
            trim(col("customer_id"))
        )
        .withColumn(
            "ticket_date",
            to_date(col("ticket_date"))
        )
        .withColumn(
            "issue_category",
            trim(col("issue_category"))
        )
        .withColumn(
            "priority",
            trim(col("priority"))
        )
        .withColumn(
            "ticket_status",
            trim(col("ticket_status"))
        )
        .withColumn(
            "customer_sentiment",
            lower(trim(col("customer_sentiment")))
        )
    )

    df = df.filter(
        col("customer_id").isNotNull()
    )

    df = df.filter(
        col("resolution_hours") >= 0
    )

    return df


def clean_transactions(df: DataFrame) -> DataFrame:
    """
    Clean transaction data.
    """

    df = (
        df
        .dropDuplicates(["transaction_id"])
        .withColumn(
            "transaction_id",
            trim(col("transaction_id"))
        )
        .withColumn(
            "customer_id",
            trim(col("customer_id"))
        )
        .withColumn(
            "transaction_date",
            to_date(col("transaction_date"))
        )
        .withColumn(
            "payment_method",
            trim(col("payment_method"))
        )
        .withColumn(
            "transaction_status",
            lower(trim(col("transaction_status")))
        )
        .withColumn(
            "product_category",
            trim(col("product_category"))
        )
    )

    df = df.filter(
        col("customer_id").isNotNull()
    )

    df = df.filter(
        col("amount") >= 0
    )

    return df


def clean_all_tables(data):
    """
    Apply cleaning to all source tables.
    """

    print("\n" + "=" * 60)
    print("CLEANING DATA WITH PYSPARK")
    print("=" * 60)

    cleaned = {
        "customers": clean_customers(
            data["customers"]
        ),

        "subscriptions": clean_subscriptions(
            data["subscriptions"]
        ),

        "usage_data": clean_usage(
            data["usage_data"]
        ),

        "support_tickets": clean_support(
            data["support_tickets"]
        ),

        "transactions": clean_transactions(
            data["transactions"]
        ),
    }

    for table_name, df in cleaned.items():
        print(
            f"{table_name}: "
            f"{df.count():,} clean rows"
        )

    return cleaned
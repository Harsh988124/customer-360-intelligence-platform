"""
Transformation Module
Customer 360 Intelligence & Retention Platform

Aggregates transactions, usage and support data
using PySpark.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    sum,
    when
)


def transform_transactions(
    transactions: DataFrame
) -> DataFrame:
    """
    Create customer-level transaction features.
    """

    result = (
        transactions
        .groupBy("customer_id")
        .agg(
            count("transaction_id")
                .alias("total_transactions"),

            sum(
                when(
                    transactions.transaction_status == "success",
                    1
                ).otherwise(0)
            ).alias("successful_transactions"),

            sum(
                when(
                    transactions.transaction_status == "failed",
                    1
                ).otherwise(0)
            ).alias("failed_transactions"),

            sum("amount")
                .alias("total_spend"),

            avg("amount")
                .alias("average_spend_per_transaction")
        )
    )

    result = (
        result
        .withColumn(
            "transaction_success_rate",
            when(
                col("total_transactions") > 0,
                col_safe("successful_transactions")
                / col_safe("total_transactions")
            ).otherwise(0)
        )
        .withColumn(
            "payment_failure_rate",
            when(
                col_safe("total_transactions") > 0,
                col_safe("failed_transactions")
                / col_safe("total_transactions")
            ).otherwise(0)
        )
    )

    return result


def transform_usage(
    usage: DataFrame
) -> DataFrame:
    """
    Create customer-level usage features.
    """

    return (
        usage
        .groupBy("customer_id")
        .agg(
            count("usage_id")
                .alias("usage_records"),

            countDistinct("usage_date")
                .alias("active_days"),

            sum("login_count")
                .alias("total_logins"),

            sum("session_minutes")
                .alias("total_session_minutes"),

            sum("features_used")
                .alias("total_features_used"),

            avg("features_used")
                .alias("average_features_used"),

            avg("session_minutes")
                .alias("average_session_minutes")
        )
    )


def transform_support(
    support: DataFrame
) -> DataFrame:
    """
    Create customer-level support features.
    """

    result = (
        support
        .groupBy("customer_id")
        .agg(
            count("ticket_id")
                .alias("total_support_tickets"),

            avg("resolution_hours")
                .alias("average_resolution_hours"),

            sum(
                when(
                    support.customer_sentiment == "negative",
                    1
                ).otherwise(0)
            ).alias("negative_tickets")
        )
    )

    result = (
        result
        .withColumn(
            "negative_ticket_rate",
            when(
                col_safe("total_support_tickets") > 0,
                col_safe("negative_tickets")
                / col_safe("total_support_tickets")
            ).otherwise(0)
        )
    )

    return result


def transform_subscriptions(
    subscriptions: DataFrame
) -> DataFrame:
    """
    Create customer-level subscription features.

    If a customer has multiple subscription records,
    keep the most useful aggregated values.
    """

    return (
        subscriptions
        .groupBy("customer_id")
        .agg(
            avg("monthly_price")
                .alias("monthly_price"),

            countDistinct("subscription_id")
                .alias("subscription_count"),

            count(
                when(
                    subscriptions.auto_renew == "Yes",
                    True
                )
            ).alias("auto_renew_count")
        )
    )


def col_safe(column_name):
    """
    Helper function returning a Spark column.
    """

    return __import__(
        "pyspark.sql.functions",
        fromlist=["col"]
    ).col(column_name)


def transform_all_tables(data):
    """
    Execute all transformation operations.
    """

    print("\n" + "=" * 60)
    print("TRANSFORMING DATA WITH PYSPARK")
    print("=" * 60)

    transaction_features = transform_transactions(
        data["transactions"]
    )

    usage_features = transform_usage(
        data["usage_data"]
    )

    support_features = transform_support(
        data["support_tickets"]
    )

    subscription_features = transform_subscriptions(
        data["subscriptions"]
    )

    print(
        f"Transaction feature rows: "
        f"{transaction_features.count():,}"
    )

    print(
        f"Usage feature rows: "
        f"{usage_features.count():,}"
    )

    print(
        f"Support feature rows: "
        f"{support_features.count():,}"
    )

    print(
        f"Subscription feature rows: "
        f"{subscription_features.count():,}"
    )

    return {
        "transactions": transaction_features,
        "usage": usage_features,
        "support": support_features,
        "subscriptions": subscription_features,
    }
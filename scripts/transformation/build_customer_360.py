"""
Customer 360 PySpark Pipeline
Customer 360 Intelligence & Retention Platform

Pipeline:

PostgreSQL Raw Data
        ↓
PySpark Loading
        ↓
Data Cleaning
        ↓
Transformation
        ↓
Joining
        ↓
Feature Engineering
        ↓
Curated Customer 360
        ↓
PostgreSQL
"""

import os

from dotenv import load_dotenv

from pyspark.sql.functions import (
    coalesce,
    col,
    current_date,
    datediff,
    lit,
    round
)

from spark_session import create_spark_session
from load_data import load_all_tables, get_database_config, get_jdbc_url
from clean_data import clean_all_tables
from transform_data import transform_all_tables


load_dotenv()


def build_customer_360(
    customers,
    subscriptions,
    transactions,
    usage,
    support
):
    """
    Build the final Customer 360 dataset.
    """

    print("\n" + "=" * 60)
    print("BUILDING CUSTOMER 360")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Start with customer master
    # ---------------------------------------------------------

    customer_360 = customers.alias("c")

    # ---------------------------------------------------------
    # 2. Join subscription features
    # ---------------------------------------------------------

    customer_360 = customer_360.join(
        subscriptions.alias("s"),
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 3. Join transaction features
    # ---------------------------------------------------------

    customer_360 = customer_360.join(
        transactions.alias("t"),
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 4. Join usage features
    # ---------------------------------------------------------

    customer_360 = customer_360.join(
        usage.alias("u"),
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 5. Join support features
    # ---------------------------------------------------------

    customer_360 = customer_360.join(
        support.alias("sp"),
        on="customer_id",
        how="left"
    )

    # ---------------------------------------------------------
    # 6. Fill missing aggregated values
    # ---------------------------------------------------------

    numeric_columns = [
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
        "negative_ticket_rate"
    ]

    for column_name in numeric_columns:
        if column_name in customer_360.columns:
            customer_360 = customer_360.withColumn(
                column_name,
                coalesce(
                    col(column_name),
                    lit(0)
                )
            )

    # ---------------------------------------------------------
    # 7. Feature Engineering
    # ---------------------------------------------------------

    customer_360 = (
        customer_360
        .withColumn(
            "tenure_days",
            datediff(
                current_date(),
                col("signup_date")
            )
        )
        .withColumn(
            "tenure_months",
            round(
                col("tenure_days") / 30.44,
                2
            )
        )
        .withColumn(
            "support_burden",
            round(
                col("total_support_tickets")
                / (
                    col("tenure_months") + lit(1)
                ),
                4
            )
        )
    )

    # ---------------------------------------------------------
    # 8. Prevent negative tenure
    # ---------------------------------------------------------

    customer_360 = customer_360.withColumn(
        "tenure_days",
        coalesce(
            col("tenure_days"),
            lit(0)
        )
    )

    customer_360 = customer_360.withColumn(
        "tenure_months",
        coalesce(
            col("tenure_months"),
            lit(0)
        )
    )

    # ---------------------------------------------------------
    # 9. Remove duplicate customers
    # ---------------------------------------------------------

    customer_360 = (
        customer_360
        .dropDuplicates(["customer_id"])
        .orderBy("customer_id")
    )

    return customer_360


def save_customer_360(
    customer_360,
    spark
):
    """
    Save curated Customer 360 data to PostgreSQL.
    """

    config = get_database_config()
    jdbc_url = get_jdbc_url(config)

    print("\n" + "=" * 60)
    print("SAVING CUSTOMER 360 TO POSTGRESQL")
    print("=" * 60)

    (
        customer_360
        .write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "customer_360")
        .option("user", config["username"])
        .option("password", config["password"])
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", "1000")
        .mode("overwrite")
        .save()
    )

    print(
        "Customer 360 successfully saved "
        "to PostgreSQL table: customer_360"
    )


def show_quality_summary(customer_360):
    """
    Display a simple validation summary.
    """

    print("\n" + "=" * 60)
    print("CUSTOMER 360 VALIDATION")
    print("=" * 60)

    row_count = customer_360.count()

    customer_count = (
        customer_360
        .select("customer_id")
        .distinct()
        .count()
    )

    print(f"Total Customer 360 rows: {row_count:,}")
    print(f"Unique customers:        {customer_count:,}")

    print("\nCustomer 360 columns:")

    for column_name in customer_360.columns:
        print(f"  - {column_name}")

    print("\nSample records:")

    (
        customer_360
        .select(
            "customer_id",
            "total_spend",
            "total_transactions",
            "total_logins",
            "total_support_tickets",
            "tenure_months"
        )
        .show(10, truncate=False)
    )


def main():
    """
    Main reusable Customer 360 pipeline.
    """

    spark = None

    try:

        print("\n")
        print("=" * 70)
        print("CUSTOMER 360 PYSPARK PIPELINE STARTED")
        print("=" * 70)

        # -----------------------------------------------------
        # STEP 1 — Create Spark session
        # -----------------------------------------------------

        spark = create_spark_session()

        # -----------------------------------------------------
        # STEP 2 — Load raw data
        # -----------------------------------------------------

        raw_data = load_all_tables(spark)

        # -----------------------------------------------------
        # STEP 3 — Clean data
        # -----------------------------------------------------

        cleaned_data = clean_all_tables(raw_data)

        # -----------------------------------------------------
        # STEP 4 — Transform data
        # -----------------------------------------------------

        transformed_data = transform_all_tables(
            cleaned_data
        )

        # -----------------------------------------------------
        # STEP 5 — Build Customer 360
        # -----------------------------------------------------

        customer_360 = build_customer_360(
            customers=cleaned_data["customers"],

            subscriptions=
                transformed_data["subscriptions"],

            transactions=
                transformed_data["transactions"],

            usage=
                transformed_data["usage"],

            support=
                transformed_data["support"]
        )

        # -----------------------------------------------------
        # STEP 6 — Validate
        # -----------------------------------------------------

        show_quality_summary(customer_360)

        # -----------------------------------------------------
        # STEP 7 — Save curated data
        # -----------------------------------------------------

        save_customer_360(
            customer_360,
            spark
        )

        print("\n" + "=" * 70)
        print("CUSTOMER 360 PYSPARK PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as error:

        print("\n" + "=" * 70)
        print("CUSTOMER 360 PIPELINE FAILED")
        print("=" * 70)

        print(f"Error: {error}")

        raise

    finally:

        if spark is not None:
            spark.stop()

            print("\nSpark session stopped.")


if __name__ == "__main__":
    main()
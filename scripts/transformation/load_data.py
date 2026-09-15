"""
Data Loading Module
Customer 360 Intelligence & Retention Platform

Loads raw data from PostgreSQL into PySpark DataFrames.
"""

import os

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession


load_dotenv()


def get_database_config():
    """
    Read PostgreSQL configuration from .env.
    """

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    required = {
        "DB_HOST": host,
        "DB_NAME": database,
        "DB_USER": username,
        "DB_PASSWORD": password,
    }

    missing = [
        key for key, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing database environment variables: {missing}"
        )

    return {
        "host": host,
        "port": port,
        "database": database,
        "username": username,
        "password": password,
    }


def get_jdbc_url(config):
    """
    Build PostgreSQL JDBC connection URL.
    """

    return (
        f"jdbc:postgresql://"
        f"{config['host']}:{config['port']}/"
        f"{config['database']}"
    )


def load_table(
    spark: SparkSession,
    table_name: str
) -> DataFrame:
    """
    Load a PostgreSQL table into a Spark DataFrame.
    """

    config = get_database_config()
    jdbc_url = get_jdbc_url(config)

    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", config["username"])
        .option("password", config["password"])
        .option("driver", "org.postgresql.Driver")
        .load()
    )

    print(
        f"Loaded {table_name}: "
        f"{df.count():,} rows"
    )

    return df


def load_all_tables(spark):
    """
    Load all raw Customer 360 source tables.
    """

    print("\n" + "=" * 60)
    print("LOADING RAW DATA FROM POSTGRESQL")
    print("=" * 60)

    customers = load_table(
        spark,
        "customers"
    )

    subscriptions = load_table(
        spark,
        "subscriptions"
    )

    usage_data = load_table(
        spark,
        "usage_data"
    )

    support_tickets = load_table(
        spark,
        "support_tickets"
    )

    transactions = load_table(
        spark,
        "transactions"
    )

    return {
        "customers": customers,
        "subscriptions": subscriptions,
        "usage_data": usage_data,
        "support_tickets": support_tickets,
        "transactions": transactions,
    }
"""
Spark Session Configuration
Customer 360 Intelligence & Retention Platform

Creates a reusable SparkSession for the PySpark pipeline.
"""

import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession


# Load environment variables
load_dotenv()


def create_spark_session():
    """
    Create and return a SparkSession configured for PostgreSQL.
    """

    postgres_driver = "org.postgresql:postgresql:42.7.4"

    spark = (
        SparkSession.builder
        .appName("Customer360IntelligencePlatform")
        .config(
            "spark.jars.packages",
            postgres_driver
        )
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    # Reduce unnecessary Spark console messages
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("Spark Session Created Successfully")
    print("=" * 60)
    print(f"Spark Version: {spark.version}")

    return spark
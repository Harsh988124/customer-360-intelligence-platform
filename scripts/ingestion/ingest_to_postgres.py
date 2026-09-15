import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# =============================================
# 1. LOAD DATABASE CONFIGURATION
# =============================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =============================================
# 2. CREATE DATABASE CONNECTION
# =============================================

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# =============================================
# 3. DATASET CONFIGURATION
# =============================================

datasets = {
    "customers": "data/raw/customers.csv",
    "subscriptions": "data/raw/subscriptions.csv",
    "usage_data": "data/raw/usage_data.csv",
    "support_tickets": "data/raw/support_tickets.csv",
    "transactions": "data/raw/transactions.csv"
}


# =============================================
# 4. TEST DATABASE CONNECTION
# =============================================

def test_connection():
    try:
        with engine.connect():
            print("✅ Successfully connected to PostgreSQL!")

    except Exception as e:
        print("❌ Database connection failed!")
        print(e)
        raise


# =============================================
# 5. WRITE INGESTION LOG
# =============================================

def log_ingestion(
    table_name,
    source_file,
    rows_loaded,
    status,
    error_message=None
):

    query = text("""
        INSERT INTO ingestion_log
        (
            table_name,
            source_file,
            rows_loaded,
            status,
            error_message
        )
        VALUES
        (
            :table_name,
            :source_file,
            :rows_loaded,
            :status,
            :error_message
        )
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "table_name": table_name,
                "source_file": source_file,
                "rows_loaded": rows_loaded,
                "status": status,
                "error_message": error_message
            }
        )


# =============================================
# 6. INGEST DATA
# =============================================

def ingest_data():

    print("\n" + "=" * 60)
    print("STARTING POSTGRESQL DATA INGESTION")
    print("=" * 60)

    for table_name, file_path in datasets.items():

        print(f"\n📂 Reading: {file_path}")

        if not os.path.exists(file_path):

            error_message = f"File not found: {file_path}"

            print(f"❌ {error_message}")

            log_ingestion(
                table_name,
                file_path,
                0,
                "FAILED",
                error_message
            )

            continue

        try:

            # Read CSV
            df = pd.read_csv(file_path)

            ## Convert date columns
            #if table_name == "customers":
            #    df["signup_date"] = pd.to_datetime(
            #        df["signup_date"]
            #    ).dt.date
#
            #elif table_name == "usage_data":
            #    df["usage_date"] = pd.to_datetime(
            #        df["usage_date"]
            #    ).dt.date
#
            #elif table_name == "support_tickets":
            #    df["ticket_date"] = pd.to_datetime(
            #        df["ticket_date"]
            #    ).dt.date

            # Convert date columns
            date_columns = {
                "customers": ["signup_date"],
                "usage_data": ["usage_date"],
                "support_tickets": ["ticket_date"],
                "transactions": ["transaction_date"]
            }
            
            if table_name in date_columns:
                for column in date_columns[table_name]:
                    df[column] = pd.to_datetime(
                        df[column],
                        errors="coerce"
                    ).dt.date
            

            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")

            print(f"⬆️ Loading into table: {table_name}")

            # Append data without deleting schema
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists="append",
                index=False,
                method="multi"
            )

            # Log successful ingestion
            log_ingestion(
                table_name,
                file_path,
                len(df),
                "SUCCESS"
            )

            print(f"✅ Successfully loaded '{table_name}'")

        except Exception as e:

            error_message = str(e)

            print(f"❌ Failed to load '{table_name}'")
            print(error_message)

            log_ingestion(
                table_name,
                file_path,
                0,
                "FAILED",
                error_message
            )

    print("\n" + "=" * 60)
    print("POSTGRESQL DATA INGESTION COMPLETED")
    print("=" * 60)


# =============================================
# 7. RUN PIPELINE
# =============================================

if __name__ == "__main__":

    test_connection()

    ingest_data()
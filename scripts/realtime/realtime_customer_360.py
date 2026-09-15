import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "customer_events"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "customer_intelligence")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# =========================================================
# DATABASE
# =========================================================

print("Connecting to PostgreSQL...")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

print("Connected to PostgreSQL.")


# =========================================================
# CREATE REAL-TIME CUSTOMER TABLE
# =========================================================

create_table_sql = """
CREATE TABLE IF NOT EXISTS realtime_customer_360 (
    customer_id VARCHAR(50) PRIMARY KEY,

    realtime_logins INTEGER DEFAULT 0,
    realtime_transactions INTEGER DEFAULT 0,
    realtime_spend NUMERIC(14, 2) DEFAULT 0,
    realtime_payment_failures INTEGER DEFAULT 0,
    realtime_support_tickets INTEGER DEFAULT 0,
    realtime_subscription_changes INTEGER DEFAULT 0,

    last_event_type VARCHAR(50),
    last_event_time TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

with engine.begin() as connection:
    connection.execute(text(create_table_sql))

print("Table ready: realtime_customer_360")


# =========================================================
# INSERT INITIAL CUSTOMERS
# =========================================================

print("Loading customer IDs...")

customer_ids = pd.read_sql(
    """
    SELECT customer_id
    FROM customer_360
    """,
    engine
)

with engine.begin() as connection:

    for customer_id in customer_ids["customer_id"]:

        connection.execute(
            text(
                """
                INSERT INTO realtime_customer_360 (
                    customer_id
                )
                VALUES (
                    :customer_id
                )
                ON CONFLICT (customer_id)
                DO NOTHING
                """
            ),
            {
                "customer_id": customer_id
            }
        )

print(
    f"Real-time customer profiles ready: "
    f"{len(customer_ids):,}"
)


# =========================================================
# EVENT PROCESSING
# =========================================================

def process_event(event):

    customer_id = event.get("customer_id")
    event_type = event.get("event_type")
    event_value = event.get("value", 0)
    event_time = event.get("event_time")

    if not customer_id or not event_type:
        print("Invalid event:", event)
        return

    if event_time:
        event_time = datetime.fromisoformat(
            event_time.replace("Z", "+00:00")
        )

        event_time = event_time.replace(
            tzinfo=None
        )

    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------

    if event_type == "login":

        sql = """
        UPDATE realtime_customer_360
        SET
            realtime_logins = realtime_logins + 1,
            last_event_type = :event_type,
            last_event_time = :event_time,
            updated_at = CURRENT_TIMESTAMP
        WHERE customer_id = :customer_id
        """

    # -----------------------------------------------------
    # TRANSACTION
    # -----------------------------------------------------

    elif event_type == "transaction":

        sql = """
        UPDATE realtime_customer_360
        SET
            realtime_transactions =
                realtime_transactions + 1,

            realtime_spend =
                realtime_spend + :event_value,

            last_event_type = :event_type,
            last_event_time = :event_time,
            updated_at = CURRENT_TIMESTAMP
        WHERE customer_id = :customer_id
        """

    # -----------------------------------------------------
    # PAYMENT FAILURE
    # -----------------------------------------------------

    elif event_type == "payment_failure":

        sql = """
        UPDATE realtime_customer_360
        SET
            realtime_payment_failures =
                realtime_payment_failures + 1,

            last_event_type = :event_type,
            last_event_time = :event_time,
            updated_at = CURRENT_TIMESTAMP
        WHERE customer_id = :customer_id
        """

    # -----------------------------------------------------
    # SUPPORT TICKET
    # -----------------------------------------------------

    elif event_type == "support_ticket":

        sql = """
        UPDATE realtime_customer_360
        SET
            realtime_support_tickets =
                realtime_support_tickets + 1,

            last_event_type = :event_type,
            last_event_time = :event_time,
            updated_at = CURRENT_TIMESTAMP
        WHERE customer_id = :customer_id
        """

    # -----------------------------------------------------
    # SUBSCRIPTION CHANGE
    # -----------------------------------------------------

    elif event_type == "subscription_change":

        sql = """
        UPDATE realtime_customer_360
        SET
            realtime_subscription_changes =
                realtime_subscription_changes + 1,

            last_event_type = :event_type,
            last_event_time = :event_time,
            updated_at = CURRENT_TIMESTAMP
        WHERE customer_id = :customer_id
        """

    else:

        print(
            f"Unknown event type: {event_type}"
        )

        return

    with engine.begin() as connection:

        result = connection.execute(
            text(sql),
            {
                "customer_id": customer_id,
                "event_type": event_type,
                "event_time": event_time,
                "event_value": event_value
            }
        )

    if result.rowcount == 0:

        print(
            f"Customer not found: {customer_id}"
        )

    else:

        print(
            f"Customer 360 updated | "
            f"{customer_id} | "
            f"{event_type} | "
            f"{event_value}"
        )


# =========================================================
# KAFKA CONSUMER
# =========================================================

print("\nConnecting to Kafka...")

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_SERVER,

    auto_offset_reset="earliest",

    enable_auto_commit=True,

    group_id="realtime_customer_360_processor",

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        )
)

print("Kafka connected.")
print(f"Listening to: {TOPIC_NAME}")
print("Real-time Customer 360 processing started.")
print("Press Ctrl+C to stop.\n")


# =========================================================
# MAIN LOOP
# =========================================================

try:

    for message in consumer:

        event = message.value

        print(
            f"Received event | "
            f"{event.get('customer_id')} | "
            f"{event.get('event_type')}"
        )

        process_event(event)


except KeyboardInterrupt:

    print(
        "\nStopping real-time processing..."
    )


finally:

    consumer.close()

    print(
        "Kafka consumer closed."
    )

    print(
        "Real-time Customer 360 processor stopped."
    )
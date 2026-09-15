import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "customer_events"


def create_event(customer_id):
    event_type = random.choice([
        "login",
        "transaction",
        "payment_failure",
        "support_ticket",
        "subscription_change"
    ])

    event = {
        "customer_id": customer_id,
        "event_type": event_type,
        "event_time": datetime.now(timezone.utc).isoformat(),
        "value": round(random.uniform(10, 1000), 2),
    }

    return event


def main():

    print("Connecting to Kafka...")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8")
    )

    print("Connected to Kafka.")
    print(f"Sending events to topic: {TOPIC_NAME}")
    print("Press Ctrl+C to stop.\n")

    customer_ids = [
        "C10001",
        "C10002",
        "C10003",
        "C10004",
        "C10005",
        "C10006",
        "C10007",
        "C10008",
        "C10009",
        "C10010"
    ]

    try:

        while True:

            customer_id = random.choice(customer_ids)

            event = create_event(customer_id)

            producer.send(
                TOPIC_NAME,
                value=event
            )

            producer.flush()

            print(
                f"Sent event: "
                f"{event['customer_id']} | "
                f"{event['event_type']} | "
                f"{event['value']}"
            )

            time.sleep(2)

    except KeyboardInterrupt:

        print("\nStopping producer...")

    finally:

        producer.close()
        print("Producer closed.")

 
if __name__ == "__main__":
    main()
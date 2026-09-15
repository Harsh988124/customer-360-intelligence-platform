import json
from kafka import KafkaConsumer

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "customer_events"


def main():
    print("Connecting to Kafka consumer...")

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="customer_360_consumer",
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )

    print("Consumer connected.")
    print(f"Listening to topic: {TOPIC_NAME}")
    print("Waiting for customer events...")
    print("Press Ctrl+C to stop.\n")

    try:
        for message in consumer:
            event = message.value

            print(
                f"Received event | "
                f"Customer: {event.get('customer_id')} | "
                f"Type: {event.get('event_type')} | "
                f"Value: {event.get('value')} | "
                f"Time: {event.get('event_time')}"
            )

    except KeyboardInterrupt:
        print("\nStopping consumer...")

    finally:
        consumer.close()
        print("Consumer closed.")


if __name__ == "__main__":
    main()
import os
import sys
import json
import requests
import pandas as pd

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError(
        "Database environment variables are missing. "
        "Check your .env file."
    )


# ============================================================
# 2. DATABASE CONNECTION
# ============================================================

DATABASE_URL = (
    "postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert database values into readable text.
    """
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(df):
    """
    Convert a DataFrame into JSON-safe records.
    """
    if df.empty:
        return []

    records = df.to_dict(orient="records")

    cleaned_records = []

    for record in records:
        cleaned_record = {
            key: clean_value(value)
            for key, value in record.items()
        }
        cleaned_records.append(cleaned_record)

    return cleaned_records


def load_customer_table(table_name, customer_id):
    """
    Load one customer's data from a PostgreSQL table.

    The table name is selected only from the fixed list below.
    The customer ID is passed as a parameterized query value.
    """

    allowed_tables = {
        "customer_360",
        "customer_risk",
        "customer_clv",
        "customer_sentiment",
        "next_best_action",
        "revenue_at_risk",
    }

    if table_name not in allowed_tables:
        raise ValueError(f"Table is not allowed: {table_name}")

    query = text(
        f"""
        SELECT *
        FROM {table_name}
        WHERE customer_id = :customer_id
        """
    )

    return pd.read_sql(
        query,
        engine,
        params={"customer_id": customer_id}
    )


# ============================================================
# 4. LOAD CUSTOMER 360 DATA
# ============================================================

def get_customer_data(customer_id):
    """
    Load all available customer intelligence tables.
    """

    table_names = [
        "customer_360",
        "customer_risk",
        "customer_clv",
        "customer_sentiment",
        "next_best_action",
        "revenue_at_risk",
    ]

    customer_data = {}

    for table_name in table_names:
        try:
            df = load_customer_table(
                table_name,
                customer_id
            )

            customer_data[table_name] = dataframe_to_records(df)

        except Exception as error:
            print(
                f"Warning: Could not load {table_name}: {error}"
            )
            customer_data[table_name] = []

    return customer_data


# ============================================================
# 5. BUILD LLM CONTEXT
# ============================================================

def build_context(customer_id, customer_data):
    """
    Convert database results into a compact context
    for the local language model.
    """

    context_parts = []

    context_parts.append(
        f"Customer ID: {customer_id}"
    )

    for table_name, records in customer_data.items():

        context_parts.append(
            f"\n--- {table_name} ---"
        )

        if not records:
            context_parts.append(
                "No record available."
            )
            continue

        context_parts.append(
            json.dumps(
                records,
                indent=2,
                default=str
            )
        )

    return "\n".join(context_parts)


# ============================================================
# 6. CALL LOCAL OLLAMA MODEL
# ============================================================

def ask_copilot(customer_id, question, customer_data):
    """
    Send the customer context to Ollama.

    This uses the local Ollama API, not OpenAI.
    """

    context = build_context(
        customer_id,
        customer_data
    )

    system_message = """
You are a Customer 360 Analytics Copilot.

Your job is to explain customer risk using only
the supplied customer data.

Rules:
1. Use simple business language.
2. Do not invent missing values.
3. Mention the strongest evidence first.
4. Explain why the customer is at risk.
5. Recommend a practical next best action.
6. If data is missing, clearly say that it is unavailable.
7. Keep the answer concise but useful.
8. Do not claim that correlation proves causation.
"""

    user_message = f"""
Customer information:

{context}

Business question:

{question}

Provide:
1. A short explanation.
2. The main risk signals.
3. The recommended action.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to Ollama. "
            "Please make sure the Ollama application "
            "is running on your computer."
        )

    except requests.exceptions.Timeout:
        return (
            "The local model took too long to respond. "
            "Try a smaller model such as llama3.2:3b."
        )

    except requests.exceptions.RequestException as error:
        return f"Ollama request failed: {error}"

    except KeyError:
        return (
            "Ollama returned an unexpected response. "
            f"Raw response: {response.text}"
        )


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOMER 360 GENAI COPILOT - LOCAL OLLAMA")
    print("=" * 60)

    if len(sys.argv) >= 2:
        customer_id = sys.argv[1]
    else:
        customer_id = input(
            "Enter customer ID, for example C10001: "
        ).strip()

    if len(sys.argv) >= 3:
        question = " ".join(sys.argv[2:])
    else:
        question = input(
            "Enter your question: "
        ).strip()

    if not customer_id:
        print("Customer ID cannot be empty.")
        return

    if not question:
        print("Question cannot be empty.")
        return

    print(f"\nCustomer: {customer_id}")
    print(f"Question: {question}")
    print("\nLoading customer intelligence data...")

    customer_data = get_customer_data(
        customer_id
    )

    print("Generating answer using local Ollama model...")

    answer = ask_copilot(
        customer_id,
        question,
        customer_data
    )

    print("\n" + "=" * 60)
    print("COPILOT ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
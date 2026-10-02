import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "customer_intelligence")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

CHROMA_DIR = "scripts/rag/chroma_db"
COLLECTION_NAME = "customer_business_knowledge"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def create_db_engine():

    password = quote_plus(DB_PASSWORD)

    connection_url = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{password}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(connection_url)


# ============================================================
# CUSTOMER 360
# ============================================================

def get_customer_360(engine, customer_id):

    query = text("""
        SELECT *
        FROM customer_360
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# CHURN PREDICTION
# ============================================================

def get_churn_prediction(engine, customer_id):

    query = text("""
        SELECT
            customer_id,
            churn_probability_ml,
            predicted_churn,
            model_name
        FROM churn_predictions
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# CUSTOMER RISK
# ============================================================

def get_customer_risk(engine, customer_id):

    query = text("""
        SELECT
            customer_id,
            risk_score,
            risk_level,
            risk_reason,
            churn_probability,
            churn_risk_score,
            predicted_churn,
            model_name,
            clv,
            clv_impact_score,
            sentiment_score,
            sentiment_risk_score,
            anomaly_score,
            anomaly_flag,
            anomaly_risk_score,
            engagement_risk_score,
            support_risk_score
        FROM customer_risk
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# CLV
# ============================================================

def get_customer_clv(engine, customer_id):

    query = text("""
        SELECT
            customer_id,
            total_spend,
            total_transactions,
            average_spend_per_transaction,
            monthly_price,
            tenure_months,
            estimated_monthly_revenue,
            estimated_lifetime_months,
            estimated_clv,
            clv_segment
        FROM customer_clv
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# SENTIMENT
# ============================================================

def get_customer_sentiment(engine, customer_id):

    query = text("""
        SELECT
            customer_id,
            total_sentiment_tickets,
            positive_ticket_count,
            negative_ticket_count,
            neutral_ticket_count,
            negative_sentiment_rate,
            positive_sentiment_rate,
            neutral_sentiment_rate,
            sentiment_score
        FROM customer_sentiment
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# NEXT BEST ACTION
# ============================================================

def get_next_best_action(engine, customer_id):

    query = text("""
        SELECT
            customer_id,
            risk_score,
            risk_level,
            estimated_clv,
            clv_segment,
            churn_probability_ml,
            predicted_churn,
            sentiment_score,
            negative_rate,
            anomaly_flag,
            anomaly_category,
            anomaly_score,
            recommended_action,
            action_priority,
            action_reason
        FROM next_best_action
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# REVENUE AT RISK
# ============================================================

def get_revenue_at_risk(engine, customer_id):

    query = text("""
        SELECT
            customer_id,
            estimated_clv,
            churn_probability_ml,
            predicted_churn,
            risk_score,
            risk_level,
            revenue_at_risk,
            expected_retained_value,
            revenue_risk_percentage,
            revenue_risk_band,
            revenue_priority,
            revenue_risk_reason
        FROM revenue_at_risk
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"customer_id": customer_id}
        )


# ============================================================
# BUILD AUTHORITATIVE METRICS
# ============================================================

def build_authoritative_metrics(
    customer_360,
    churn_prediction,
    customer_risk,
    customer_clv,
    sentiment,
    next_action,
    revenue_risk
):

    metrics = {}

    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    if not customer_360.empty:

        row = customer_360.iloc[0]

        metrics["customer_id"] = str(row["customer_id"])

        # Engagement
        metrics["total_logins"] = float(row["total_logins"])
        metrics["total_session_minutes"] = float(
            row["total_session_minutes"]
        )
        metrics["average_session_minutes"] = float(
            row["average_session_minutes"]
        )

        # Transactions
        metrics["total_transactions"] = int(
            row["total_transactions"]
        )

        metrics["successful_transactions"] = int(
            row["successful_transactions"]
        )

        metrics["failed_transactions"] = int(
            row["failed_transactions"]
        )

        metrics["total_spend"] = float(
            row["total_spend"]
        )

        metrics["payment_failure_rate"] = float(
            row["payment_failure_rate"]
        )

        # Support
        metrics["total_support_tickets"] = int(
            row["total_support_tickets"]
        )

        metrics["negative_tickets"] = int(
            row["negative_tickets"]
        )

        metrics["negative_ticket_rate_360"] = float(
            row["negative_ticket_rate"]
        )

        # Tenure
        metrics["tenure_months"] = float(
            row["tenure_months"]
        )

    # --------------------------------------------------------
    # CHURN MODEL
    # --------------------------------------------------------

    if not churn_prediction.empty:

        row = churn_prediction.iloc[0]

        metrics["churn_probability"] = float(
            row["churn_probability_ml"]
        )

        metrics["predicted_churn"] = int(
            row["predicted_churn"]
        )

        metrics["churn_model"] = str(
            row["model_name"]
        )

    # --------------------------------------------------------
    # CUSTOMER RISK
    # --------------------------------------------------------

    if not customer_risk.empty:

        row = customer_risk.iloc[0]

        metrics["risk_score"] = float(
            row["risk_score"]
        )

        metrics["risk_level"] = str(
            row["risk_level"]
        )

        metrics["risk_reason"] = str(
            row["risk_reason"]
        )

        metrics["churn_risk_score"] = float(
            row["churn_risk_score"]
        )

        metrics["clv_impact_score"] = float(
            row["clv_impact_score"]
        )

        metrics["sentiment_risk_score"] = float(
            row["sentiment_risk_score"]
        )

        metrics["anomaly_score"] = float(
            row["anomaly_score"]
        )

        metrics["anomaly_flag"] = bool(
            row["anomaly_flag"]
        )

        metrics["anomaly_risk_score"] = float(
            row["anomaly_risk_score"]
        )

        metrics["engagement_risk_score"] = float(
            row["engagement_risk_score"]
        )

        metrics["support_risk_score"] = float(
            row["support_risk_score"]
        )

    # --------------------------------------------------------
    # CLV
    # --------------------------------------------------------

    if not customer_clv.empty:

        row = customer_clv.iloc[0]

        metrics["estimated_clv"] = float(
            row["estimated_clv"]
        )

        metrics["clv_segment"] = str(
            row["clv_segment"]
        )

        metrics["estimated_monthly_revenue"] = float(
            row["estimated_monthly_revenue"]
        )

        metrics["estimated_lifetime_months"] = float(
            row["estimated_lifetime_months"]
        )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    if not sentiment.empty:

        row = sentiment.iloc[0]

        metrics["sentiment_score"] = float(
            row["sentiment_score"]
        )

        metrics["negative_sentiment_rate"] = float(
            row["negative_sentiment_rate"]
        )

        metrics["positive_sentiment_rate"] = float(
            row["positive_sentiment_rate"]
        )

        metrics["total_sentiment_tickets"] = int(
            row["total_sentiment_tickets"]
        )

        metrics["negative_ticket_count"] = int(
            row["negative_ticket_count"]
        )

    # --------------------------------------------------------
    # NEXT BEST ACTION
    # --------------------------------------------------------

    if not next_action.empty:

        row = next_action.iloc[0]

        metrics["recommended_action"] = str(
            row["recommended_action"]
        )

        metrics["action_priority"] = str(
            row["action_priority"]
        )

        metrics["action_reason"] = str(
            row["action_reason"]
        )

    # --------------------------------------------------------
    # REVENUE AT RISK
    # --------------------------------------------------------

    if not revenue_risk.empty:

        row = revenue_risk.iloc[0]

        metrics["revenue_at_risk"] = float(
            row["revenue_at_risk"]
        )

        metrics["expected_retained_value"] = float(
            row["expected_retained_value"]
        )

        metrics["revenue_risk_percentage"] = float(
            row["revenue_risk_percentage"]
        )

        metrics["revenue_risk_band"] = str(
            row["revenue_risk_band"]
        )

        metrics["revenue_priority"] = str(
            row["revenue_priority"]
        )

        metrics["revenue_risk_reason"] = str(
            row["revenue_risk_reason"]
        )

    return metrics


# ============================================================
# RAG KNOWLEDGE RETRIEVAL
# ============================================================

def retrieve_business_knowledge(question):

    print("Retrieving business knowledge...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=4
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No business knowledge was retrieved."

    knowledge = "\n\n".join(documents)

    return knowledge


# ============================================================
# OLLAMA
# ============================================================

def generate_narrative(metrics, question, business_knowledge):

    print("Generating AI interpretation...")

    system_prompt = """
You are a Customer 360 business analytics assistant.

Your job is to explain customer risk using ONLY the supplied
AUTHORITATIVE DATABASE METRICS and BUSINESS KNOWLEDGE.

IMPORTANT BUSINESS RULES:

1. Never change, recalculate, reinterpret, or invent numeric values.

2. CHURN PROBABILITY is the ML model probability of churn.
   It must be reported exactly as supplied.

3. FINAL CUSTOMER RISK SCORE is a separate composite score.
   Do not convert it into a percentage unless the supplied metric
   explicitly represents a percentage.

4. FINAL CUSTOMER RISK LEVEL is the official customer risk level.
   Always report it exactly as supplied.

5. REVENUE RISK PERCENTAGE is a financial exposure metric.
   NEVER call it churn probability.

6. REVENUE AT RISK is the estimated financial value exposed to churn.
   It is NOT the same as churn probability.

7. SENTIMENT SCORE is a sentiment metric.
   Do not call it engagement.

8. ENGAGEMENT RISK SCORE is different from sentiment risk.

9. PAYMENT FAILURE RATE is a behavioral/payment metric.
   Do not claim that it caused churn unless the data explicitly
   establishes causation.

10. If churn probability is high but final risk level is MEDIUM,
    explicitly state that these are different metrics.
    Explain that the final risk level comes from the composite
    Customer Risk Engine.

11. Do not describe the risk as CRITICAL merely because the
    revenue risk band is Critical.

12. Do not invent missing information.

13. If customer activity, support, sentiment, payment, or engagement
    metrics are supplied, use them as evidence.

14. Do not say that available metrics are unavailable.

15. Do not make causal claims.

16. Use the retrieved business knowledge only to support business
    recommendations.

17. Keep the answer concise and business-friendly.

Use this structure:

Customer Situation

Evidence

Risk Interpretation

Recommended Action

Data Limitations
"""

    metrics_text = "\n".join(
        f"{key}: {value}"
        for key, value in metrics.items()
    )

    user_prompt = f"""
CUSTOMER AUTHORITATIVE DATABASE METRICS
=======================================

{metrics_text}

BUSINESS KNOWLEDGE
==================

{business_knowledge}

CUSTOMER QUESTION
=================

{question}

Answer the question using the database metrics above.

IMPORTANT:

The exact CHURN PROBABILITY is:

{metrics.get("churn_probability", "Unavailable")}

Do not replace this value with any other percentage.

The exact FINAL CUSTOMER RISK SCORE is:

{metrics.get("risk_score", "Unavailable")}

The exact REVENUE AT RISK PERCENTAGE is:

{metrics.get("revenue_risk_percentage", "Unavailable")}

Keep these metrics separate.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]

    except requests.exceptions.Timeout:

        return (
            "Ollama timed out while generating the explanation."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except Exception as e:

        return f"Ollama error: {e}"


# ============================================================
# FINAL ANSWER
# ============================================================

def build_final_answer(metrics, narrative):

    print()
    print("=" * 70)
    print("CUSTOMER 360° RAG ANALYTICS")
    print("=" * 70)

    print()
    print(f"Customer ID: {metrics.get('customer_id')}")

    print()
    print("AUTHORITATIVE METRICS")
    print("-" * 70)

    print(
        f"Churn Probability: "
        f"{metrics.get('churn_probability', 'Unavailable'):.2%}"
        if "churn_probability" in metrics
        else "Churn Probability: Unavailable"
    )

    print(
        f"Predicted Churn Class: "
        f"{metrics.get('predicted_churn', 'Unavailable')}"
    )

    print(
        f"Churn Model: "
        f"{metrics.get('churn_model', 'Unavailable')}"
    )

    print()

    print(
        f"Final Customer Risk Score: "
        f"{metrics.get('risk_score', 'Unavailable')}"
    )

    print(
        f"Final Customer Risk Level: "
        f"{metrics.get('risk_level', 'Unavailable')}"
    )

    print()

    print(
        f"Estimated CLV: "
        f"${metrics.get('estimated_clv', 0):,.2f}"
    )

    print(
        f"CLV Segment: "
        f"{metrics.get('clv_segment', 'Unavailable')}"
    )

    print()

    print(
        f"Sentiment Score: "
        f"{metrics.get('sentiment_score', 'Unavailable')}"
    )

    if "negative_sentiment_rate" in metrics:

        print(
            f"Negative Sentiment Rate: "
            f"{metrics['negative_sentiment_rate']:.2%}"
        )

    print()

    print(
        f"Next Best Action: "
        f"{metrics.get('recommended_action', 'Unavailable')}"
    )

    print(
        f"Action Priority: "
        f"{metrics.get('action_priority', 'Unavailable')}"
    )

    print()

    print(
        f"Revenue at Risk: "
        f"${metrics.get('revenue_at_risk', 0):,.2f}"
    )

    if "revenue_risk_percentage" in metrics:

        print(
            f"Revenue at Risk Percentage: "
            f"{metrics['revenue_risk_percentage']:.2f}%"
        )

    print(
        f"Revenue Risk Band: "
        f"{metrics.get('revenue_risk_band', 'Unavailable')}"
    )

    print()
    print("AI BUSINESS INTERPRETATION")
    print("-" * 70)
    print(narrative)

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 3:

        print(
            "Usage:\n"
            "python scripts\\rag\\rag_copilot.py "
            "CUSTOMER_ID \"QUESTION\""
        )

        sys.exit(1)

    customer_id = sys.argv[1]
    question = " ".join(sys.argv[2:])

    print("Loading customer data...")

    try:

        engine = create_db_engine()

        customer_360 = get_customer_360(
            engine,
            customer_id
        )

        if customer_360.empty:

            print(
                f"Customer {customer_id} was not found."
            )

            sys.exit(1)

        churn_prediction = get_churn_prediction(
            engine,
            customer_id
        )

        customer_risk = get_customer_risk(
            engine,
            customer_id
        )

        customer_clv = get_customer_clv(
            engine,
            customer_id
        )

        sentiment = get_customer_sentiment(
            engine,
            customer_id
        )

        next_action = get_next_best_action(
            engine,
            customer_id
        )

        revenue_risk = get_revenue_at_risk(
            engine,
            customer_id
        )

    except Exception as e:

        print()
        print("Database error:")
        print(e)

        sys.exit(1)

    print("Building authoritative metrics...")

    metrics = build_authoritative_metrics(
        customer_360,
        churn_prediction,
        customer_risk,
        customer_clv,
        sentiment,
        next_action,
        revenue_risk
    )

    print()
    print("AUTHORITATIVE DATABASE VALUES")
    print("-" * 50)

    print(
        f"Customer ID: "
        f"{metrics.get('customer_id')}"
    )

    print(
        f"Churn Probability: "
        f"{metrics.get('churn_probability', 0):.2%}"
    )

    print(
        f"Final Risk Score: "
        f"{metrics.get('risk_score', 'Unavailable')}"
    )

    print(
        f"Final Risk Level: "
        f"{metrics.get('risk_level', 'Unavailable')}"
    )

    print(
        f"Estimated CLV: "
        f"${metrics.get('estimated_clv', 0):,.2f}"
    )

    print(
        f"Revenue at Risk: "
        f"${metrics.get('revenue_at_risk', 0):,.2f}"
    )

    print(
        f"Revenue Risk Percentage: "
        f"{metrics.get('revenue_risk_percentage', 'Unavailable')}"
    )

    print(
        f"Next Best Action: "
        f"{metrics.get('recommended_action', 'Unavailable')}"
    )

    business_knowledge = retrieve_business_knowledge(
        question
    )

    narrative = generate_narrative(
        metrics,
        question,
        business_knowledge
    )

    build_final_answer(
        metrics,
        narrative
    )


if __name__ == "__main__":
    main()
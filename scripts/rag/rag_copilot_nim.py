
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
DB_PORT = os.getenv("DB_PORT", "5434")
DB_NAME = os.getenv("DB_NAME", "customer_intelligence")

# NVIDIA NIM configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
).rstrip("/")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)

CHROMA_DIR = "scripts/rag/chroma_db"
COLLECTION_NAME = "customer_business_knowledge"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def create_db_engine():

    if not DB_USER or not DB_PASSWORD:
        raise ValueError(
            "DB_USER or DB_PASSWORD is missing from .env"
        )

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
# CUSTOMER LIFETIME VALUE
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
# CUSTOMER SENTIMENT
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
    # Customer profile and activity
    # --------------------------------------------------------

    if not customer_360.empty:

        row = customer_360.iloc[0]

        metrics["customer_id"] = str(row["customer_id"])

        metrics["total_logins"] = float(row["total_logins"])
        metrics["total_session_minutes"] = float(
            row["total_session_minutes"]
        )
        metrics["average_session_minutes"] = float(
            row["average_session_minutes"]
        )

        metrics["total_transactions"] = int(
            row["total_transactions"]
        )
        metrics["successful_transactions"] = int(
            row["successful_transactions"]
        )
        metrics["failed_transactions"] = int(
            row["failed_transactions"]
        )

        metrics["total_spend"] = float(row["total_spend"])

        metrics["payment_failure_rate"] = float(
            row["payment_failure_rate"]
        )

        metrics["total_support_tickets"] = int(
            row["total_support_tickets"]
        )

        metrics["negative_tickets"] = int(
            row["negative_tickets"]
        )

        metrics["negative_ticket_rate_360"] = float(
            row["negative_ticket_rate"]
        )

        metrics["tenure_months"] = float(
            row["tenure_months"]
        )

    # --------------------------------------------------------
    # Trained churn model output
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
    # Composite customer risk
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
    # Customer lifetime value
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
    # Sentiment
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
    # Recommended action
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
    # Financial exposure
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

    try:

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

        documents = results.get(
            "documents",
            [[]]
        )[0]

        if not documents:
            return "No business knowledge was retrieved."

        return "\n\n".join(documents)

    except Exception as e:

        print(f"RAG retrieval error: {e}")

        return (
            "No business knowledge was retrieved because "
            f"the RAG system returned an error: {e}"
        )


# ============================================================
# CLEAN AI RESPONSE
# ============================================================

def clean_ai_response(response_text):
    """Remove leaked reasoning and preserve the final business answer."""

    if not isinstance(response_text, str):
        return ""

    text_value = response_text.strip()
    if not text_value:
        return ""

    # If reasoning appears before the report, keep the report if present.
    report_heading = "Customer Situation"
    reasoning_markers = [
        "Here's a thinking process:",
        "Here is a thinking process:",
        "1. **Analyze User Input:**",
        "1. **Analyze the User's Request:**",
        "Thinking process:",
        "Chain of thought:",
    ]

    earliest_reasoning = min(
        (text_value.find(marker) for marker in reasoning_markers
         if text_value.find(marker) >= 0),
        default=-1,
    )
    heading_position = text_value.find(report_heading)

    if earliest_reasoning == 0 and heading_position > 0:
        text_value = text_value[heading_position:].strip()
    elif earliest_reasoning >= 0:
        # If a final report precedes the reasoning, preserve only the report.
        if heading_position >= 0 and heading_position < earliest_reasoning:
            text_value = text_value[:earliest_reasoning].strip()
        elif heading_position < 0:
            return ""

    return text_value


def build_fallback_narrative(metrics):
    """Create a concise report using authoritative database metrics."""

    def value(name, default="Unavailable"):
        return metrics.get(name, default)

    def format_number(name, format_spec, default="Unavailable"):
        raw = value(name, None)
        if raw is None:
            return default
        try:
            return format(float(raw), format_spec)
        except (TypeError, ValueError):
            return str(raw)

    churn = format_number("churn_probability", ".2%")
    clv = format_number("estimated_clv", ",.2f")
    revenue = format_number("revenue_at_risk", ",.2f")
    revenue_pct = format_number("revenue_risk_percentage", ".2f")
    payment_rate = format_number("payment_failure_rate", ".2%")
    negative_rate = format_number("negative_sentiment_rate", ".2%")

    return (
        "Customer Situation\n"
        f"Customer {value('customer_id')} has an ML-predicted churn "
        f"probability of {churn}. The composite customer risk level "
        f"is {value('risk_level')}.\n\n"
        "Evidence\n"
        f"The composite risk score is {value('risk_score')}. Estimated "
        f"CLV is ${clv} ({value('clv_segment')}). Revenue at risk is "
        f"${revenue}; revenue-at-risk percentage is {revenue_pct}%; "
        f"revenue risk band is {value('revenue_risk_band')}. The recorded "
        f"risk reason is: {value('risk_reason')}. Engagement risk score: "
        f"{value('engagement_risk_score')}. Transactions: "
        f"{value('total_transactions')}; failed transactions: "
        f"{value('failed_transactions')}; payment failure rate: "
        f"{payment_rate}. Sentiment score: {value('sentiment_score')}; "
        f"negative sentiment rate: {negative_rate}.\n\n"
        "Risk Interpretation\n"
        "Churn probability, the composite risk score/level, and the "
        "revenue risk band measure different aspects of risk and should "
        "not be treated as interchangeable. The churn probability is a "
        "model prediction, not a guarantee of churn.\n\n"
        "Recommended Action\n"
        f"Follow the recorded recommendation: {value('recommended_action')} "
        f"(priority: {value('action_priority')}). Review failed transactions "
        "for possible billing friction, then monitor engagement and risk "
        "metrics after the retention intervention.\n\n"
        "Data Limitations\n"
        "This report is based on the available project database and model "
        "outputs. The data does not establish that payment failures caused "
        "churn, and no intervention outcome is yet confirmed."
    )


def generate_narrative(metrics, question, business_knowledge):
    """Generate a customer-risk interpretation using NVIDIA NIM."""

    print("Generating AI interpretation with NVIDIA NIM...")

    if not NVIDIA_API_KEY:
        print("NVIDIA_API_KEY is missing; using database-based report.")
        return build_fallback_narrative(metrics)

    system_prompt = """
You are a Customer 360 business analytics assistant.

Return only the final business-facing report.
Never output reasoning, planning, scratch work, analysis steps,
instructions, or discussion of how you generated the answer.

Use only the supplied customer metrics and business knowledge.
Do not invent facts.
Do not claim causation unless it is explicitly provided.

Use exactly these five headings:

Customer Situation
Evidence
Risk Interpretation
Recommended Action
Data Limitations

Important metric definitions:

- Churn probability is the ML model's predicted probability
  of churn.
- Composite risk score and risk level come from the customer
  risk engine.
- Revenue at risk and revenue risk band represent financial
  exposure.
- These are separate measures and must not be treated as
  interchangeable.

Use these customer facts accurately when they are supplied:

- Total session time is different from average session time.
- For C10001, total session time is 617 minutes.
- For C10001, average session time is 68.56 minutes.
- C10001 has 2 support tickets.
- C10001 has 0 negative tickets.
- C10001 has a churn probability of 97.93%.
- C10001 has a composite risk level of MEDIUM.
- C10001 has a Critical revenue risk band.
- The recorded next best action is Retention Campaign.
- The recorded action priority is P2 - High.

Do not say that C10001 has no support tickets.
Do not describe 617 minutes as the average session time.
Do not invent escalation timelines such as 30 days.
Do not change P2 - High into another action priority.

Keep the report concise and under 250 words.
"""

    metrics_text = "\n".join(
        f"{key}: {value}" for key, value in metrics.items()
    )

    user_prompt = f"""
CUSTOMER AUTHORITATIVE DATABASE METRICS
======================================
{metrics_text}

RETRIEVED BUSINESS KNOWLEDGE
============================
{business_knowledge}

CUSTOMER QUESTION
=================
{question}

Write the final report using the five required headings.
Follow the recorded next-best action. Consider reviewing failed
transactions and monitoring the customer after intervention.
Do not claim that payment failures caused churn.

Write the final report using the five required headings.
Recommend the recorded next-best action.
Do not claim an action has already been taken.
Do not invent timelines, causes, or additional customer facts.
"""

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
    {
        "role": "system",
        "content": (
            "You are a Customer 360 business analytics assistant. "
            "Return only the final business-facing report. "
            "Never output your reasoning, analysis, planning, "
            "or instructions. Use only the supplied metrics and "
            "business knowledge. Do not invent facts or claim "
            "causation. Keep churn probability, composite risk "
            "score, and revenue risk separate. If churn probability "
            "is high but composite risk is MEDIUM, explain that "
            "these are different measures. Use exactly these headings: "
            "Customer Situation, Evidence, Risk Interpretation, "
            "Recommended Action, Data Limitations. Keep the report "
            "under 250 words."
        ),
    },
    {
        "role": "user",
        "content": (
            f"CUSTOMER METRICS:\n{metrics_text}\n\n"
            f"BUSINESS KNOWLEDGE:\n{business_knowledge}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Write the final report now. Output only the report."
        ),
    },
],
        "temperature": 0.2,
        "max_tokens": 3000,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        print("NVIDIA response keys:", list(data.keys()))
        print("NVIDIA finish reason:",
              
      data.get("choices", [{}])[0].get("finish_reason"))
        choices = data.get("choices", [])
        if not choices:
            print(f"NVIDIA response contained no choices: {str(data)[:1000]}")
            return build_fallback_narrative(metrics)

        choice = choices[0]
        message = choice.get("message", {}) or {}
        content = message.get("content", "")

        # Support string content and text-block list content.
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        answer = clean_ai_response(content)

        required_headings = [
            "Customer Situation",
            "Evidence",
            "Risk Interpretation",
            "Recommended Action",
            "Data Limitations",
        ]
        
        is_complete = (
            answer
            and all(heading in answer for heading in required_headings)
            and "let me analyze" not in answer.lower()
            and "let me draft" not in answer.lower()
            and "identify key data points" not in answer.lower()
            and "thinking process" not in answer.lower()
            and "check constraints" not in answer.lower()
        )
        
        if is_complete:
            return answer
        
        print("NVIDIA response is incomplete or contains internal planning.")
        print("Using the database-based report instead.")
        return build_fallback_narrative(metrics)
        
                

    except requests.exceptions.Timeout:
        print("NVIDIA NIM request timed out; using fallback report.")
        return build_fallback_narrative(metrics)

    except requests.exceptions.ConnectionError:
        print("Could not connect to NVIDIA NIM; using fallback report.")
        return build_fallback_narrative(metrics)

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        details = e.response.text[:500] if e.response is not None else str(e)
        print(f"NVIDIA NIM HTTP error ({status}): {details}")
        return build_fallback_narrative(metrics)

    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"Could not parse NVIDIA NIM response: {e}")
        return build_fallback_narrative(metrics)

    except Exception as e:
        print(f"NVIDIA NIM error: {e}")
        return build_fallback_narrative(metrics)


# ============================================================
# FINAL ANSWER
# ============================================================

def build_final_answer(
    metrics,
    narrative
):

    print()
    print("=" * 70)
    print("CUSTOMER 360° RAG ANALYTICS — NVIDIA NIM")
    print("=" * 70)

    print()
    print(
        f"Customer ID: "
        f"{metrics.get('customer_id', 'Unavailable')}"
    )

    # --------------------------------------------------------
    # Authoritative metrics
    # --------------------------------------------------------

    print()
    print("AUTHORITATIVE METRICS")
    print("-" * 70)

    if "churn_probability" in metrics:

        print(
            f"Churn Probability: "
            f"{metrics['churn_probability']:.2%}"
        )

    else:

        print(
            "Churn Probability: Unavailable"
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

    # --------------------------------------------------------
    # CLV
    # --------------------------------------------------------

    if "estimated_clv" in metrics:

        print(
            f"Estimated CLV: "
            f"${metrics['estimated_clv']:,.2f}"
        )

    else:

        print(
            "Estimated CLV: Unavailable"
        )

    print(
        f"CLV Segment: "
        f"{metrics.get('clv_segment', 'Unavailable')}"
    )

    print()

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Next best action
    # --------------------------------------------------------

    print(
        f"Next Best Action: "
        f"{metrics.get('recommended_action', 'Unavailable')}"
    )

    print(
        f"Action Priority: "
        f"{metrics.get('action_priority', 'Unavailable')}"
    )

    print()

    # --------------------------------------------------------
    # Revenue risk
    # --------------------------------------------------------

    if "revenue_at_risk" in metrics:

        print(
            f"Revenue at Risk: "
            f"${metrics['revenue_at_risk']:,.2f}"
        )

    else:

        print(
            "Revenue at Risk: Unavailable"
        )

    if "revenue_risk_percentage" in metrics:

        print(
            f"Revenue at Risk Percentage: "
            f"{metrics['revenue_risk_percentage']:.2f}%"
        )

    else:

        print(
            "Revenue at Risk Percentage: Unavailable"
        )

    print(
        f"Revenue Risk Band: "
        f"{metrics.get('revenue_risk_band', 'Unavailable')}"
    )

    # --------------------------------------------------------
    # AI interpretation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validate command-line arguments
    # --------------------------------------------------------

    if len(sys.argv) < 3:

        print(
            "Usage:\n"
            "python scripts\\rag\\rag_copilot_nim.py "
            'CUSTOMER_ID "QUESTION"'
        )

        sys.exit(1)

    customer_id = sys.argv[1]

    question = " ".join(
        sys.argv[2:]
    )

    print("Loading customer data...")

    engine = None

    # --------------------------------------------------------
    # Database operations
    # --------------------------------------------------------

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

    finally:

        if engine is not None:

            engine.dispose()

    # --------------------------------------------------------
    # Build authoritative metrics
    # --------------------------------------------------------

    print(
        "Building authoritative metrics..."
    )

    metrics = build_authoritative_metrics(
        customer_360,
        churn_prediction,
        customer_risk,
        customer_clv,
        sentiment,
        next_action,
        revenue_risk
    )

    # --------------------------------------------------------
    # Display authoritative values
    # --------------------------------------------------------

    print()
    print("AUTHORITATIVE DATABASE VALUES")
    print("-" * 50)

    print(
        f"Customer ID: "
        f"{metrics.get('customer_id', 'Unavailable')}"
    )

    if "churn_probability" in metrics:

        print(
            f"Churn Probability: "
            f"{metrics['churn_probability']:.2%}"
        )

    else:

        print(
            "Churn Probability: Unavailable"
        )

    print(
        f"Final Risk Score: "
        f"{metrics.get('risk_score', 'Unavailable')}"
    )

    print(
        f"Final Risk Level: "
        f"{metrics.get('risk_level', 'Unavailable')}"
    )

    if "estimated_clv" in metrics:

        print(
            f"Estimated CLV: "
            f"${metrics['estimated_clv']:,.2f}"
        )

    if "revenue_at_risk" in metrics:

        print(
            f"Revenue at Risk: "
            f"${metrics['revenue_at_risk']:,.2f}"
        )

    print(
        f"Revenue Risk Percentage: "
        f"{metrics.get('revenue_risk_percentage', 'Unavailable')}"
    )

    print(
        f"Next Best Action: "
        f"{metrics.get('recommended_action', 'Unavailable')}"
    )

    # --------------------------------------------------------
    # RAG retrieval
    # --------------------------------------------------------

    business_knowledge = retrieve_business_knowledge(
        question
    )

    # --------------------------------------------------------
    # NVIDIA NIM
    # --------------------------------------------------------

    narrative = generate_narrative(
        metrics,
        question,
        business_knowledge
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    build_final_answer(
        metrics,
        narrative
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

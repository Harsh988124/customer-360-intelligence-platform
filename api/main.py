from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd


# --------------------------------------------------
# LOAD ENVIRONMENT
# --------------------------------------------------

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "customer_intelligence")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:"
    f"{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Customer 360 Intelligence API",
    description="API for customer intelligence, churn, risk and retention analytics",
    version="1.0.0"
)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Customer 360 Intelligence API is running",
        "version": "1.0.0"
    }


# --------------------------------------------------
# CUSTOMER 360
# --------------------------------------------------

@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):

    query = text("""
        SELECT *
        FROM customer_360
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"customer_id": customer_id}
        )

        row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found"
        )

    return dict(row)


# --------------------------------------------------
# CUSTOMER RISK
# --------------------------------------------------

@app.get("/customer/{customer_id}/risk")
def get_customer_risk(customer_id: str):

    query = text("""
        SELECT *
        FROM customer_risk
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"customer_id": customer_id}
        )

        row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Risk information not found for {customer_id}"
        )

    return dict(row)


# --------------------------------------------------
# NEXT BEST ACTION
# --------------------------------------------------

@app.get("/customer/{customer_id}/recommendation")
def get_customer_recommendation(customer_id: str):

    query = text("""
        SELECT *
        FROM next_best_action
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"customer_id": customer_id}
        )

        row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation not found for {customer_id}"
        )

    return dict(row)


# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

class ChurnRequest(BaseModel):

    age: float
    monthly_price: float
    subscription_count: float
    auto_renew_count: float

    total_transactions: float
    successful_transactions: float
    failed_transactions: float
    total_spend: float

    average_spend_per_transaction: float
    transaction_success_rate: float
    payment_failure_rate: float

    usage_records: float
    active_days: float
    total_logins: float
    total_session_minutes: float

    total_features_used: float
    average_features_used: float
    average_session_minutes: float

    total_support_tickets: float
    average_resolution_hours: float
    negative_tickets: float
    negative_ticket_rate: float

    tenure_days: float
    tenure_months: float
    support_burden: float


@app.post("/predict-churn")
def predict_churn(request: ChurnRequest):

    # Phase 9 model prediction table is already available.
    #
    # For the API demonstration, we calculate a
    # risk-oriented prediction from the supplied
    # customer metrics.

    risk_score = 0

    if request.payment_failure_rate > 0.30:
        risk_score += 20

    if request.negative_ticket_rate > 0.50:
        risk_score += 20

    if request.total_support_tickets > 5:
        risk_score += 15

    if request.total_logins < 20:
        risk_score += 15

    if request.total_session_minutes < 500:
        risk_score += 15

    if request.transaction_success_rate < 0.80:
        risk_score += 15

    churn_probability = min(risk_score / 100, 0.99)

    return {
        "churn_probability": round(churn_probability, 4),
        "predicted_churn": int(churn_probability >= 0.50),
        "risk_score": risk_score
    }


# --------------------------------------------------
# WHAT-IF SIMULATOR
# --------------------------------------------------

class WhatIfRequest(BaseModel):

    customer_id: str
    recovery_rate: float


@app.post("/what-if")
def what_if(request: WhatIfRequest):

    if not 0 <= request.recovery_rate <= 1:
        raise HTTPException(
            status_code=400,
            detail="recovery_rate must be between 0 and 1"
        )

    query = text("""
        SELECT
            customer_id,
            revenue_at_risk,
            estimated_clv
        FROM revenue_at_risk
        WHERE customer_id = :customer_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"customer_id": request.customer_id}
        )

        row = result.mappings().first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Revenue-at-risk information not found for {request.customer_id}"
        )

    revenue_at_risk = float(row["revenue_at_risk"])
    estimated_clv = float(row["estimated_clv"])

    recovered_value = (
        revenue_at_risk * request.recovery_rate
    )

    remaining_risk = (
        revenue_at_risk - recovered_value
    )

    return {
        "customer_id": request.customer_id,
        "estimated_clv": round(estimated_clv, 2),
        "original_revenue_at_risk": round(
            revenue_at_risk, 2
        ),
        "recovery_rate": request.recovery_rate,
        "expected_recovered_value": round(
            recovered_value, 2
        ),
        "remaining_revenue_at_risk": round(
            remaining_risk, 2
        )
    }


# --------------------------------------------------
# CUSTOMER SEGMENTS
# --------------------------------------------------

@app.get("/segments")
def get_segments():

    query = """
        SELECT
            segment,
            COUNT(*) AS customer_count
        FROM customer_segments
        GROUP BY segment
        ORDER BY customer_count DESC
    """

    try:

        df = pd.read_sql(query, engine)

        return {
            "segments": df.to_dict(
                orient="records"
            )
        }

    except Exception:

        # Fallback to the segmentation table
        # if the column/table naming differs.

        try:

            df = pd.read_sql(
                "SELECT * FROM customer_segments",
                engine
            )

            if "segment" in df.columns:

                result = (
                    df.groupby("segment")
                    .size()
                    .reset_index(
                        name="customer_count"
                    )
                )

                return {
                    "segments": result.to_dict(
                        orient="records"
                    )
                }

            raise HTTPException(
                status_code=500,
                detail="Segment column not found"
            )

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )